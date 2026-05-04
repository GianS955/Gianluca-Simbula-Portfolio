from network import Network

class Agent:
    def __init__(self, infos):
        if infos.get('Actor Network',None) is None:
            raise KeyError(f'No informations on \"Actor Network\" were passed.')
        self.actor_network = Network(infos['Actor Network'])

        if infos.get('Critic Network',None) is None:
            raise KeyError(f'No informations on \"Critic Network\" were passed.')
        self.critic_network = Network(infos['Critic Network'])