"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: PID algorithm definitions. Slight deviation from PEP convention: action variables are defined in capitals.
Modified by: Javier Belmonte
Date: 04/26/2020
Added: Fuzzy PID control class and rules
"""

from constants import *
import abc
import json
import matplotlib.pyplot as mpl
import numpy as np
from skfuzzy import interp_membership, defuzz
from skfuzzy.membership import (gaussmf, gauss2mf, gbellmf, piecemf, pimf,
                                psigmf, sigmf, smf, trapmf, trimf, zmf)


class PID():
    """ Called from the children of PID_Framework"""
    def __init__(self, Kp, Ki, Kd, C=None, P=None, D=None, rules= None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.accumulated_error = 0
        self.P = 0
        self.D = 0
        self.max_value = 0
        self.min_value = 0

        # Try to load config
        if rules is not None:
            with open(rules) as jsonfile:
                config = json.loads(jsonfile.read())
                P = config['P']
                D = config['D']
                C = config['C']

        # Error
        if P:
            self.p_range = np.arange(*P['range'])
            self.p_mf = self.generate_mf_group(P['memberships'], self.p_range)
        else:
            self.p_range = None

        # Derivative error
        if D:
            self.d_range = np.arange(*D['range'])
            self.d_mf = self.generate_mf_group(D['memberships'], self.d_range)
        else:
            self.d_range = None

        if C:
            # Fuzzy Class
            self.c_range = np.arange(*C['range'])
            self.c_mf = self.generate_mf_group(C['memberships'], self.c_range)
            self.c_rules = C['rules']
        else:
            self.c_range = None

    def generate_mf_group(self, G, x):
        mf_group = {}
        for (k, v) in G.items():
            shp = v['shp']
            mf = v['mf']
            if mf == 'trap':
                mf_group[k] = trapmf(x, shp)
            if mf == 'tri':
                mf_group[k] = trimf(x, shp)
            if mf == 'gbell':
                mf_group[k] = gbellmf(x, shp[0], shp[1], shp[2])
            if mf == 'gauss':
                mf_group[k] = gaussmf(x, shp[0], shp[1])
            if mf == 'gauss2':
                mf_group[k] = gauss2mf(x, shp[0], shp[1])
            if mf == 'sig':
                mf_group[k] = sigmf(x, shp[0], shp[1])
            if mf == 'psig':
                mf_group[k] = psigmf(x, shp[0], shp[1], shp[2], shp[3])
            if mf == 'zmf':
                mf_group[k] = zmf(x, shp[0], shp[1], shp[2], shp[3])
            if mf == 'smf':
                mf_group[k] = smf(x, shp[0], shp[1], shp[2], shp[3])
            if mf == 'pimf':
                mf_group[k] = pimf(x, shp[0], shp[1], shp[2], shp[3])
            if mf == 'piecemf':
                mf_group[k] = piecemf(x, shp[0], shp[1], shp[2], shp[3])
        return mf_group

    def show_mf_groups(self):
        if self.p_range is not None:
            mpl.subplot(3, 1, 1)
            for label, mf in self.p_mf.items():
                mpl.plot(mf)
        if self.d_range is not None:
            mpl.subplot(3, 1, 2)
            for label, mf in self.d_mf.items():
                mpl.plot(mf)
        if self.c_range is not None:
            mpl.subplot(3, 1, 3)
            for label, mf in self.c_mf.items():
                mpl.plot(mf)
        mpl.show()

    def calculate(self, p, d):
        """ Calculates the fuzzy output gain """

        # Calculate membership value for each function
        if self.p_range is not None:
            p_interp = {k: interp_membership(self.p_range, mf, p) for k, mf in self.p_mf.items()}
            #print(max(p_interp.items(), key=operator.itemgetter(1))[0])
        else:
            p_interp = {}
        if self.d_range is not None:
            d_interp = {k: interp_membership(self.d_range, mf, d) for k, mf in self.d_mf.items()}
            #print(max(d_interp.items(), key=operator.itemgetter(1))[0])
        else:
            d_interp = {}

        # Merge rule-bases
        dicts = [p_interp, d_interp]
        super_dict = {}
        for k in set(k for d in dicts for k in d):
            super_dict[k] = [d[k] for d in dicts if k in d]

        # Generated inferences by rule implications
        aggregate_membership = np.zeros(len(self.c_range))
        for a, b, c in self.c_rules:
            try:
                impl = np.fmin(super_dict[a], super_dict[b]) * self.c_mf[c]
                aggregate_membership = np.fmax(impl, aggregate_membership)
            except:
                pass

        c = defuzz(self.c_range, aggregate_membership, 'centroid')
        return c  # this is the resulting "value" of the current state

    def increment_intregral_error(self, error, pi_limit=3):
        self.accumulated_error = self.accumulated_error + error
        if self.accumulated_error > pi_limit:
            self.accumulated_error = pi_limit
        elif self.accumulated_error < pi_limit:
            self.accumulated_error = -pi_limit

    def compute_output(self, error, dt_error):
        self.increment_intregral_error(error)
        return self.Kp * error + self.Ki * self.accumulated_error + self.Kd * dt_error

    def compute_output_f(self, error, dt_error):
        self.increment_intregral_error(error)
        fuzzy = self.calculate(error,dt_error)
        return fuzzy * self.Kp * error + self.Ki * self.accumulated_error + self.Kd * dt_error * fuzzy


class PID_Framework():
    """ Sets the skeleton code for the actual pid algorithms (children) to inherit. """

    @abc.abstractmethod
    def pid_algorithm(self, s, x_target, y_target):
        pass


class PID_Benchmark(PID_Framework):
    """ Tuned PID Benchmark against which all other algorithms are compared. """

    def __init__(self):
        super(PID_Benchmark, self).__init__()
        self.Fe_PID = PID(10, 0, 10)
        self.psi_PID = PID(0.085, 0.001, 10.55)
        self.Fs_theta_PID = PID(5, 0, 6)

    def pid_algorithm(self, s, x_target=None, y_target=None):
        dx, dy, vel_x, vel_y, theta, omega, legContact_left, legContact_right = s
        if x_target is not None:
            dx = dx - x_target
        if y_target is not None:
            dy = dy - y_target
        # ------------------------------------------
        y_ref = -0.1  # Adjust speed
        y_error = y_ref - dy + 0.1 * dx
        y_dterror = -vel_y + 0.1 * vel_x

        Fe = self.Fe_PID.compute_output(y_error, y_dterror) * (abs(dx) * 50 + 1)
        # ------------------------------------------
        theta_ref = 0
        theta_error = theta_ref - theta + 0.2 * dx  # theta is negative when slanted to the north east
        theta_dterror = -omega + 0.2 * vel_x
        Fs_theta = self.Fs_theta_PID.compute_output(theta_error, theta_dterror)
        Fs = -Fs_theta  # + Fs_x
        # ------------------------------------------
        theta_ref = 0
        theta_error = -theta_ref + theta
        theta_dterror = omega
        if abs(dx) > 0.01 and dy < 0.5:
            theta_error = theta_error - 0.06 * dx  # theta is negative when slanted to the right
            theta_dterror = theta_dterror - 0.06 * vel_x
        psi = self.psi_PID.compute_output(theta_error, theta_dterror)

        if legContact_left and legContact_right:  # legs have contact
            Fe = 0
            Fs = 0

        return Fe, Fs, psi


class PID_psi(PID_Framework):
    """ PID for controlling just the angle of the rocket nozzle. """

    def __init__(self):
        super(PID_psi, self).__init__()
        self.psi = PID(0.1, 0, 0.01)

    def pid_algorithm(self, s, x_target=None, y_target=None):
        dx, dy, vel_x, vel_y, theta, omega, legContact_left, legContact_right = s
        theta_error = theta
        theta_dterror = -omega - vel_x
        psi = self.psi.compute_output(theta_error, theta_dterror)
        return psi


class Fuzzy_PID(PID_Framework):
    """ Fuzzy PID algorithm """

    def __init__(self):
        super(Fuzzy_PID, self).__init__()
        self.Fe_PID = PID(10, 0, 10, rules='rules_Fe.json')
        self.psi_PID = PID(0.085, 0.001, 10.55, rules='rules_psi.json')
        self.Fs_theta_PID = PID(5, 0, 6, rules='rules_Fs.json')

    def pid_algorithm(self, s, x_target=None, y_target=None):
        dx, dy, vel_x, vel_y, theta, omega, legContact_left, legContact_right = s
        if x_target is not None:
            dx = dx - x_target
        if y_target is not None:
            dy = dy - y_target
        # ------------------------------------------
        y_ref = -0.1  # Adjust speed
        y_error = y_ref - dy + 0.1 * dx
        y_dterror = -vel_y + 0.1 * vel_x

        Fe = self.Fe_PID.compute_output_f(y_error, y_dterror) * (abs(dx) * 50 + 1)
        # ------------------------------------------
        theta_ref = 0
        theta_error = theta_ref - theta + 0.2 * dx  # theta is negative when slanted to the north east
        theta_dterror = -omega + 0.2 * vel_x

        Fs_theta = self.Fs_theta_PID.compute_output_f(theta_error, theta_dterror)
        Fs = -Fs_theta  # + Fs_x
        # ------------------------------------------
        theta_ref = 0
        theta_error = -theta_ref + theta
        theta_dterror = omega
        if abs(dx) > 0.01 and dy < 0.5:
            theta_error = theta_error - 0.06 * dx  # theta is negative when slanted to the right
            theta_dterror = theta_dterror - 0.06 * vel_x

        psi = self.psi_PID.compute_output_f(theta_error, theta_dterror)

        if legContact_left and legContact_right:  # legs have contact
            Fe = 0
            Fs = 0

        return Fe, Fs, psi
