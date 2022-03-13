import numpy as np
from FiniteMDPClass import FiniteMDPClass


class CustomerSupportMDPClass(FiniteMDPClass):

    def __init__(self):
        self._NumState = 4  # [None, Low, Medium, High]
        self._NumAction = 2  # [Support, Charge]

        # #################################### TRANSACTIONS #################################
        self._PCharge = np.zeros((self._NumState, self._NumState))  # Probabilities for Charge
        self._PSupport = np.zeros((self._NumState, self._NumState))  # Probabilities for Support

        # State Low and action Charge
        self._PCharge[1, 0] = 0.6
        self._PCharge[1, 1] = 0.4

        # State Medium and action Charge
        self._PCharge[2, 1] = 0.7
        self._PCharge[2, 2] = 0.3

        # State High and action Charge
        self._PCharge[3, 2] = 0.6
        self._PCharge[3, 3] = 0.4

        # State None and action Support
        self._PSupport[0, 1] = 0.9
        self._PSupport[0, 0] = 0.1

        # State Low and action Support
        self._PSupport[1, 2] = 0.6
        self._PSupport[1, 1] = 0.4

        # State Medium and action Support
        self._PSupport[2, 3] = 0.75
        self._PSupport[2, 2] = 0.25

        # State High and action Support
        self._PSupport[3, 3] = 0.9
        self._PSupport[3, 2] = 0.1

        # with 2 actions: Charge and Support

        self._P = np.zeros((self._NumState, self._NumState, self._NumAction))
        self._P[:, :, 0] = self._PCharge
        self._P[:, :, 1] = self._PSupport

        # #################################### REWARDS #################################
        self._R = np.zeros((self._NumState, self._NumAction))

        # State Empty and action Re-breed
        self._R[0, 1] = -2500
       
        # State Low and action Fish
        self._R[1, 0] = 100
        self._R[2, 0] = 200
        self._R[3, 0] = 500
        # #################################### ACTIONS #################################s
        self._A = np.ones((self._NumState, self._NumAction))
    

        FiniteMDPClass(self._P, self._R, self._A)
