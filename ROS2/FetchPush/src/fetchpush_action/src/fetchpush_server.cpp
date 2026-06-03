#include <memory>
#include <functional>
#include <thread>
#include <cmath>

#include <rclcpp/rclcpp.hpp>
#include <torch/script.h>
#include "rclcpp_action/rclcpp_action.hpp"
#include "fetchpush_action_interface/action/fetch_push.hpp"
#include "fetchpush_msgs/msg/observation.hpp"
#include "fetchpush_msgs/msg/action.hpp"
#include "fetchpush_msgs/msg/desired_goal.hpp"


namespace fetchpush_action{
  class FetchPushActionServer : public rclcpp::Node
  {
    public:
      using FetchPushInterface = fetchpush_action_interface::action::FetchPush;
      using FetchPushGoalHandle = rclcpp_action::ServerGoalHandle<FetchPushInterface>;
    
      explicit FetchPushActionServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions()) : Node("fetchpush_action_server", options), episode_done_(false){
        using namespace std::placeholders;
    
        auto handle_goal= [this](const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const FetchPushInterface::Goal> goal){
          RCLCPP_INFO(this->get_logger(), "Recieved goal request");
          (void)uuid; // TO AVOID COMPILER WARNING "UNUSED PARAMETER UUID"
          (void)goal;
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        };
    
        auto handle_cancel = [this](const std::shared_ptr<FetchPushGoalHandle> goal_handle){
          RCLCPP_INFO(this->get_logger(), "Recieved request to cancel goal");
          (void) goal_handle;
          return rclcpp_action::CancelResponse::ACCEPT;
        };
    
        auto handle_accepted = [this](const std::shared_ptr<FetchPushGoalHandle> goal_handle){
          auto execute_in_thread = [this, goal_handle](){return this->execute(goal_handle);};
          std::thread{execute_in_thread}.detach();
        };
    
        this->action_server_ = rclcpp_action::create_server<FetchPushInterface>(
          this,
          "FetchPush",
          handle_goal,
          handle_cancel,
          handle_accepted
        );      

        this->declare_parameter("model_path", "");
        std::string model_path = this->get_parameter("model_path").as_string();
        if(model_path.empty()){
          RCLCPP_ERROR(this->get_logger(), "Parameter 'model_path' is empty");
          throw std::runtime_error("model_path parameter is required");
        }

        try{
          actor_ = torch::jit::load(model_path);
          actor_.eval();
        } catch(const c10::Error & e){
          RCLCPP_ERROR(this->get_logger(), "Failed to load model: %s", e.what());
          throw;
        }

        goal_publisher_ = this->create_publisher<fetchpush_msgs::msg::DesiredGoal>("desired_goal",10);
        action_publisher_ = this->create_publisher<fetchpush_msgs::msg::Action>("action", 10);
        feedback_ = std::make_shared<FetchPushInterface::Feedback>();

        observation_subscriber_ = this->create_subscription<fetchpush_msgs::msg::Observation>(
          "observation", 10,
          [this](const fetchpush_msgs::msg::Observation::SharedPtr msg){
            this->observation_callback(msg);
          }
        );

      }    


    private:

      std::shared_ptr<FetchPushGoalHandle> goal_handle_;
      rclcpp_action::Server<FetchPushInterface>::SharedPtr action_server_;
      torch::jit::script::Module actor_;
      rclcpp::Publisher<fetchpush_msgs::msg::Action>::SharedPtr action_publisher_;
      rclcpp::Publisher<fetchpush_msgs::msg::DesiredGoal>::SharedPtr goal_publisher_;
      rclcpp::Subscription<fetchpush_msgs::msg::Observation>::SharedPtr observation_subscriber_; 

      std::shared_ptr<FetchPushInterface::Feedback> feedback_;
      std::atomic<bool> episode_done_;

      void execute(const std::shared_ptr<FetchPushGoalHandle> goal_handle){
        RCLCPP_INFO(this->get_logger(), "Executing goal");
        episode_done_ = false;
        goal_handle_ = goal_handle;
        rclcpp::Rate loop_rate(1);
        const auto goal = goal_handle->get_goal();
        
        int & current_step = feedback_->current_step;
        current_step = 0;
        

        auto goal_msg = fetchpush_msgs::msg::DesiredGoal();
        goal_msg.desired_goal = goal->desired_goal;
        goal_msg.goal_id = rclcpp_action::to_string(goal_handle->get_goal_id());
        this->goal_publisher_->publish(goal_msg);

        episode_done_ = false;

        while (!episode_done_ && rclcpp::ok()){
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
  
      void observation_callback(const fetchpush_msgs::msg::Observation::SharedPtr msg){

        if (msg->terminal)
        {
          RCLCPP_INFO(this->get_logger(), "Episode successfully concluded");
          
          auto result = std::make_shared<FetchPushInterface::Result>();
          bool & success = result->success;
          success = true;
          int & total_steps = result->total_steps;
          total_steps = feedback_->current_step; 

          episode_done_ = true;
          goal_handle_->succeed(result);
          return;
        }

        if (msg->truncated){
          RCLCPP_INFO(this->get_logger(), "Episode not concluded");
          
          auto result = std::make_shared<FetchPushInterface::Result>();
          bool & success = result->success;
          success = false;
          int & total_steps = result->total_steps;
          total_steps = feedback_->current_step; 

          episode_done_ = true;
          goal_handle_->abort(result);
          return;
        }

        std::vector<float> input_vec;
        input_vec.insert(input_vec.end(), msg->observation.begin(), msg->observation.end());
        input_vec.insert(input_vec.end(), msg->desired_goal.begin(), msg->desired_goal.end());

        torch::Tensor input = torch::tensor(input_vec).unsqueeze(0);
        std::vector<torch::jit::IValue> inputs = {input};
        auto output = actor_.forward(inputs).toTuple();
        torch::Tensor action = output->elements()[0].toTensor();

        auto action_msg = fetchpush_msgs::msg::Action();
        for (size_t i = 0; i < action_msg.action.size(); i++){
          action_msg.action[i] = action[0][i].item<float>();
        }
        
        // PUBLICATION OF FEEDBACK
        float & distance = feedback_->distance;
        distance = 0.0;
        for (size_t i=0; i< msg->desired_goal.size(); i++){
          distance = distance + pow((msg->desired_goal[i] - msg->achieved_goal[i]),2);
        }
        distance = sqrt(distance);        
        
        int & current_step = feedback_->current_step;
        current_step = current_step + 1;
        
        goal_handle_->publish_feedback(feedback_);
        RCLCPP_INFO(this->get_logger(), "Feedback published, step: %d, distance: %f", feedback_->current_step, feedback_->distance);

        this->action_publisher_->publish(action_msg);
      }

    };

}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<fetchpush_action::FetchPushActionServer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
