import collections
import os

import numpy as np
from myosuite.envs.myo.myobase import register_env_with_variants
from myosuite.envs.myo.myobase.walk_v0 import WalkEnvV0


class WalkEnvCustomRewardV0(WalkEnvV0):
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "gaussian_vel": 5,
        "grf": -0.07281,
        "smooth_exc": -0.097,
        "number_muscles": -1.57929,
        "joint_limit": -0.1307,
    }

    def step(self, *args, **kwargs):
        self._prev_ctrl = self.sim.data.ctrl.copy()
        return super().step(*args, **kwargs)

    def _gaussian_plateau_vel(self):
        # TODO: account for x_velocity properly. cp mujoco which allows drift. Reward should prefer model to stay on straight line not slowly drift sidwways.
        # TODO: this keeps the velocity reward at zero until model is moving quite near target. fine for low targets. not for high targets.
        x_vel, y_vel = self._get_com_velocity()

        return np.exp(
            -np.square(x_vel),
        ) + np.exp(
            -np.square(y_vel - self.target_y_vel),
        )

    def _grf(self):
        # TODO: calculate and store weight early on. In __init?
        # TODO: convert the magic 1.2 into a parameter, not least because running would occur higher GRFs
        # TODO: find data on GRFs for walking/running gaits.
        r_grf = (
            self.sim.data.sensor("r_foot").data[0]
            + self.sim.data.sensor("r_toes").data[0]
        )
        l_grf = (
            self.sim.data.sensor("l_foot").data[0]
            + self.sim.data.sensor("l_toes").data[0]
        )
        weight = 9.8 * sum(self.sim.model.body_mass)
        # the feet and toe sensors are <touch> sensors which return a single scalar value for
        # surface forces acting through the touch "site" along a normal to the contacting surface.
        # At least I think that's what they do.
        # Either way the values are in Newtons. We normalized this against the weight so the
        # normalized_grf is in units of body weight "BW" (which mirrors how Scone returns contact_load)
        normalized_grf = (r_grf + l_grf) / weight
        # and then return this value clipped below 1.2 - a magic number from the original paper which
        # serves to avoid any penalty for grfs which would occur in normal walking.
        return max(0, normalized_grf - 1.2)

    def _exc_smooth_cost(self):
        # ctrl is the excitation array
        # act is the resulting activation state
        # actuator_force is the resulting force
        delta_excs = self.sim.data.ctrl - self._prev_ctrl
        return np.mean(np.square(delta_excs))

    def _number_muscle_cost(self):
        # TODO: convert the magic 0.15 number to a parameter. May want to tweak it for running?
        return self._get_proportion_active_muscles(0.15)

    def _get_proportion_active_muscles(self, threshold):
        """
        Get the proportion of muscles whose activations are above a threshold.
        """
        return (
            np.count_nonzero(self.sim.data.act > threshold)
            / self.sim.data.act.size
        )

    def _joint_limit_torques(self):
        # Use the efc arrays directly. These contain the details of the currently active
        # constraints. I think. We need to do this to extract the joint limit constraints
        # and thus the forces/torques imposed by those constraints.
        #
        # Have found that the number of joint limit constraints this finds is always the
        # same number of joints that are out of their defined ranges. Which is encouraging.
        #
        # Use efc_type array to select the constraints that are joint-limits. And use
        # that to select from the actual efc_force array. However, this will return
        # values in N for slide joints and Nm for hinge joints. So summing them is not
        # technically a valid thing to do, and as far as I can tell, the Scone
        # implementation only cares about torques. So, we can use the efc_id array to
        # find the joint ids, use that to index the jnt_type array, and disambiguate the
        # hinges from the slides.
        #
        # For now we only deal with the torques (from hinge joints) per the Scone
        # implementation. Adding support for slide limit forces is left as a todo.
        # TODO: also support slide joint limit force cost.
        joint_limit_constraint_idx = np.nonzero(
            self.sim.data.efc_type
            == self.sim.lib.mjtConstraint.mjCNSTR_LIMIT_JOINT
        )
        jnt_forces = self.sim.data.efc_force[joint_limit_constraint_idx]
        jnt_ids = self.sim.data.efc_id[joint_limit_constraint_idx]
        jnt_types = self.sim.model.jnt_type[jnt_ids]
        sum_hinge_torques = np.sum(
            jnt_forces[
                np.nonzero(jnt_types == self.sim.lib.mjtJoint.mjJNT_HINGE)
            ]
        )

        # Scone implementation returns a mean average across all axes and all
        # joints. Which, I think in MuJoCo land means divide by the number of hinge
        # joints as each hinge in MuJoCo only has one axis (in Scone it looks like a
        # single joint incorporates all 3 axes).
        num_hinge_joints = np.count_nonzero(
            self.sim.model.jnt_type == self.sim.lib.mjtJoint.mjJNT_HINGE
        )

        return sum_hinge_torques / num_hinge_joints

    def get_reward_dict(self, obs_dict):
        vel_reward = self._get_vel_reward()

        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys
                # TODO: name these better
                ("gaussian_vel", self._gaussian_plateau_vel()),
                ("grf", self._grf()),
                ("smooth_exc", self._exc_smooth_cost()),
                ("number_muscles", self._number_muscle_cost()),
                ("joint_limit", self._joint_limit_torques()),
                # Must keys
                ("sparse", vel_reward),
                ("solved", vel_reward >= 1.0),
                ("done", self._get_done()),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()],
            axis=0,
        )
        return rwd_dict


register_env_with_variants(
    id="myoLegWalk-v0-customReward",
    entry_point="deprl.custom_myosuite:WalkEnvCustomRewardV0",
    max_episode_steps=1000,
    kwargs={
        "model_path": os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../myo_sim/leg/myolegs_basic_scene.xml",
        ),
        "normalize_act": True,
        "min_height": 0.8,  # minimum center of mass height before reset
        "max_rot": 0.8,  # maximum rotation before reset
        "hip_period": 100,  # desired periodic hip angle movement
        "reset_type": "init",  # none, init, random
        "target_x_vel": 0.0,  # desired x velocity in m/s
        "target_y_vel": 1.2,  # desired y velocity in m/s
        "target_rot": None,  # if None then the initial root pos will be taken, otherwise provide quat
        "weighted_reward_keys": WalkEnvCustomRewardV0.DEFAULT_RWD_KEYS_AND_WEIGHTS,
    },
)
