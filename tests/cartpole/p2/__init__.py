from tests.cartpole import CartPoles

class DuoCartPoles(CartPoles):
    def __init__(self):
        super(DuoCartPoles, self).__init__(n_players=2)
