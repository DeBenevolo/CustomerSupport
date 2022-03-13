import numpy as np
from FiniteMDPClass import FiniteMDPClass


class SalmonMDPClass(FiniteMDPClass):

    def __init__(self):
        self._NumState = 4  # [Empty, Low, Medium, High]
        self._NumAction = 3  # [Fish, Not_to_fish, Re-breed]

        # #################################### TRANSACTIONS #################################
        self._PFish = np.zeros((self._NumState, self._NumState))  # Probabilities for Fish
        self._PNotFish = np.zeros((self._NumState, self._NumState))  # Probabilities for Not to Fish
        self._PReBreed = np.zeros((self._NumState, self._NumState))  # Probabilities for Re-breed

        # State Low and action Fish
        self._PFish[1, 0] = 0.75
        self._PFish[1, 1] = 0.25

        # State Medium and action Fish
        self._PFish[2, 1] = 0.75
        self._PFish[2, 2] = 0.25

        # State High and action Fish
        self._PFish[3, 2] = 0.6
        self._PFish[3, 3] = 0.4

        # State Low and action Not_to_fish
        self._PNotFish[1, 2] = 0.7
        self._PNotFish[1, 1] = 0.3

        # State Medium and action Not_to_fish
        self._PNotFish[2, 3] = 0.75
        self._PFish[2, 2] = 0.25

        # State High and action Not_to_fish
        self._PNotFish[3, 3] = 0.95
        self._PNotFish[3, 2] = 0.05

        # State Empty and action Re-breed
        self._PReBreed[0, 1] = 1

        # with 3 actions: Fish, Not_to_fish and Re-breed

        self._P = np.zeros((self._NumState, self._NumState, self._NumAction))
        self._P[:, :, 0] = self._PFish
        self._P[:, :, 1] = self._PNotFish
        self._P[:, :, 2] = self._PReBreed

        # #################################### REWARDS #################################
        self._R = np.zeros((self._NumState, self._NumAction))

        # State Empty and action Re-breed
        self._R[0, 2] = -200
        self._R[0, 0] = -10000
        self._R[0, 1] = -10000

        # State Low and action Fish
        self._R[1, 0] = 5
        self._R[2, 0] = 10
        self._R[3, 0] = 50
        # #################################### ACTIONS #################################

        self._A = np.ones((self._NumState, self._NumAction))
    

        FiniteMDPClass(self._P, self._R, self._A)
