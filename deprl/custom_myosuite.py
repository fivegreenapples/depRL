import collections
import math
import os
import time

import numpy as np
from myosuite.envs.myo.myobase import register_env_with_variants
from myosuite.envs.myo.myobase.walk_v0 import WalkEnvV0
from myosuite.utils.quat_math import quat2mat


class WalkEnvCustomRewardV0(WalkEnvV0):
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "gaussian_vel_x": 2,
        "gaussian_vel_y": 5,
        "grf": -1,
        "smooth_exc": -12.5,
        "number_muscles": -0.5,
        "joint_limit": -0.03,
        "forward_lean": 0.5,
        "sideways_lean": 0.5,
        "forward_direction": 0.2,
        "done": -1000,
    }

    def _setup(
        self,
        curriculum=None,
        **kwargs,
    ):
        self.curriculum = curriculum
        super()._setup(**kwargs)

    def step(self, *args, **kwargs):
        self._prev_ctrl = self.sim.data.ctrl.copy()
        if self.curriculum:
            if self.curriculum["type"] == "random":
                if self.steps % self.curriculum["change_steps"] == 0:
                    # calc new target
                    self.target_y_vel = self.curriculum["vmin"] + (
                        np.random.random()
                        * (self.curriculum["vmax"] - self.curriculum["vmin"])
                    )
                    # print(
                    #     f"New target y vel: {self.target_y_vel:.2f} m/s"
                    #     f" {self.target_y_vel*3.6:.1f} kph"
                    # )
            elif self.curriculum["type"] == "accelerate":
                if self.steps % self.curriculum["change_steps"] == 0:
                    self.target_y_vel = self.curriculum["vmin"] + (
                        (self.steps // self.curriculum["change_steps"])
                        * self.curriculum["inc"]
                    )
                    if self.target_y_vel > self.curriculum["vmax"]:
                        self.target_y_vel = self.curriculum["vmax"]
                    # print(
                    #     f"New target y vel: {self.target_y_vel:.2f} m/s"
                    #     f" {self.target_y_vel*3.6:.1f} kph"
                    # )
            else:
                raise ValueError(
                    f"Unhandled curriculum type: '{self.curriculum['type']}'"
                )

        return super().step(*args, **kwargs)

    def _orientation(self):
        # Here we establish the forward lean, sideways lean, and deviation from
        # forward direction, returning them separately to allow for varying weights.
        #
        # Note for reasons I don't quite understand the y and x axes seem to work in
        # the opposite directions to what I would expect. That is, if the model moves
        # forward along positive y-direction (confirmed by velocity), and z-axis is
        # positive up (my assumption) then the calculation of forward lean should
        # result in positive y. But I get a negative result. Similarly for deviation
        # from forward direction, I get a negative result (which seems to imply the
        # model's local coordinates have y-axis positive out its back. Kinda weird and
        # no doubt something I'm misunderstanding. Either way the use of negative unit
        # vectors below is to get positive results in directions I want them.

        # Get the rotation matrix from the quaternion
        quat = self.sim.data.qpos[3:7].copy()
        rot_mat = quat2mat(quat)

        # Establish forward and sideways components of z-vector for torso lean amount.
        # See above for why negative unit vector
        # x is sideways, positive fall to right, 0 means no lean
        x_component_of_unit_z = -rot_mat[0][2]  # aka (rot_mat @ [0, 0, -1])[0]
        # y is forward/back, positive fall to front, 0 means no lean
        y_component_of_unit_z = -rot_mat[1][2]  # aka (rot_mat @ [0, 0, -1])[1]

        # The amount the body is facing in the direction of travel
        # local x coord of model faces forward, but model is made to move along the y-axis
        # of environment. Kind of bonkers but seems to be true.
        # Positive faces forward. Fully forward would be 1.
        # Using negative unit vector per above comment.
        y_component_of_unit_x = -rot_mat[1][0]  # aka (rot_mat @ [-1, 0, 0])[1]

        # for walking we want basically upright, but for running a small forward lean
        # is expected (5-7 degrees). So that we don't prefer upright, we'll allow a
        # forward lean of 0 - 10 degrees without penalty.
        # TODO: find reference for forward lean for running
        forward_lean_angle = (180 / math.pi) * math.asin(y_component_of_unit_z)
        if forward_lean_angle <= 0:
            # leaning backwards, punish steeply
            forward_lean_reward = (20 + forward_lean_angle) / 20
        elif forward_lean_angle <= 10:
            # allowed lean amount
            forward_lean_reward = 1
        else:
            # leaning forwards, punish as increased lean
            forward_lean_reward = (40 - forward_lean_angle) / 30
        forward_lean_reward = max(0, min(1, forward_lean_reward))

        # sideways lean is treated symmetrically
        sideays_lean_angle = (180 / math.pi) * math.asin(x_component_of_unit_z)
        sideways_lean_reward = (20 - abs(sideays_lean_angle)) / 20
        sideways_lean_reward = max(0, min(1, sideways_lean_reward))

        # Some forward deviation may be acceptable owing to hip swing but straight ahead
        # is still preferred on average. We'll just raise the deviation value to a high
        # power so going straight ahead scores 1 and drops off steeply either side.
        # TODO: look up hip swing
        # Above did not work quite correctly, leaving comment for reminder of history.
        # The issue is that humans tend to keep their torso pointing forward while the
        # hip swings as we walk. But in our simulation the torso and hips are locked
        # together so we must allow the torso to swing because the hips do so. The
        # resulting movement is of course affected by the weight applied to this reward
        # term. In a previous run the weight favored hips with zero swing and led to an
        # interesting gait of the legs.
        # So new strategy is to allow a 15 degree swing either way without punishment.
        # cosine(15deg) == 0.966
        # Shift reward up so +/-15 degrees is above 1
        forward_direction_reward = y_component_of_unit_x + (1 - 0.966)
        # Apply high power to punish strongly either side of 15 degrees.
        forward_direction_reward = y_component_of_unit_x**8
        # Clip 0 - 1
        forward_direction_reward = max(0, min(1, forward_direction_reward))

        return (
            forward_lean_reward,
            sideways_lean_reward,
            forward_direction_reward,
        )

    def _gaussian_vel(self):
        # TODO: account for sideways drift. Reward should prefer model to stay on straight line not slowly drift sideways. Might be better as a separate position reward not vel reward.
        x_vel, y_vel = self._get_com_velocity()

        # Velocity reward broken into reward from sideways movement (always targetting 0 m/s)
        # and reward from forward movement (according to externally set preference)
        # For forward we use an asymetric curve around the target so the reward shape
        # encourages gradually getting up to target speed, but penalises going over.

        # For sideways movement, use a narrow gaussian to make transverse velocity undesirable
        x_reward = np.exp(-np.square(5 * x_vel))

        if self.target_y_vel > 0 and y_vel <= self.target_y_vel:
            # For forward velocities less than target, scale gaussian according to target
            # so reward goes from approx 0 at zero velocity to 1 at target
            y_reward = np.exp(
                -np.square(
                    (2 / self.target_y_vel) * (y_vel - self.target_y_vel)
                )
            )
        else:
            # For forward velocities above target apply a steepish curve to
            # disincentivise going over.
            y_reward = np.exp(-np.square(3 * (y_vel - self.target_y_vel)))

        return x_reward, y_reward

    def _grf(self, threshold):
        # TODO: calculate and store weight early on. In __init?
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
        # and then return this value clipped below a threshold - a magic number from the original paper which
        # serves to avoid any penalty for grfs which would occur in normal walking.
        return max(0, normalized_grf - threshold)

    def _exc_smooth_cost(self, normalized_speed):
        # ctrl is the excitation array
        # act is the resulting activation state
        # actuator_force is the resulting force
        delta_excs = self.sim.data.ctrl - self._prev_ctrl
        # see elsewhere for the magic 1.24 number
        return np.mean(np.square(delta_excs)) / (normalized_speed**1.24)

    def _number_muscle_cost(self, threshold):
        return self._get_proportion_active_muscles(threshold)

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
        forward_lean, sideways_lean, forward_direction = self._orientation()

        vel_x_score, vel_y_score = self._gaussian_vel()

        # adjust threshold used for proportion of muscles above a certain activation
        # threshold. asumption is faster target velocities require more active muscles.
        # Normalise by walking speed and multiply magic number (from paper) by it.
        target_velocity_normed_by_walking_speed = self.target_y_vel / 1.2
        # normalized velocity taken to the power of 1.24 gets us to 0.70 activation
        # threshold at 15kph
        muscle_threshold = 0.15 * (
            target_velocity_normed_by_walking_speed**1.24
        )

        # Flag to indicate if walker has fallen vertically. Use to end the simulation.
        # Avoids learning collapse when walker just kneels down.
        too_low = self._get_height() < self.min_height
        # TODO add condition about overally moved distance. e.g. not moved expected
        # amount in certain time. Basically recognising a velocity collapse.

        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys
                # TODO: name these better
                ("gaussian_vel_x", vel_x_score),
                ("gaussian_vel_y", vel_y_score),
                # this multiplier is pretty generous for running. human grfs should be pretty well below this.
                # allows penalty to be high here if we want.
                (
                    "grf",
                    self._grf(1.2 * target_velocity_normed_by_walking_speed),
                ),
                (
                    "smooth_exc",
                    self._exc_smooth_cost(
                        target_velocity_normed_by_walking_speed
                    ),
                ),
                ("number_muscles", self._number_muscle_cost(muscle_threshold)),
                ("joint_limit", self._joint_limit_torques()),
                ("forward_lean", forward_lean),
                ("sideways_lean", sideways_lean),
                ("forward_direction", forward_direction),
                # Must keys
                ("sparse", vel_reward),
                ("solved", vel_reward >= 1.0),
                (
                    "done",
                    too_low
                    or forward_lean == 0
                    or sideways_lean == 0
                    or forward_direction == 0,
                ),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()],
            axis=0,
        )
        return rwd_dict

    # This is a copy from the base class WalkEnvV0 to make it easier to insert the target
    # velocity observations before the act array. Need act to stay at the end to satisfy
    # expectations of the custom replay buffer AdaptiveEnergyBuffer.
    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict["t"] = np.array([sim.data.time])
        obs_dict["time"] = np.array([sim.data.time])
        obs_dict["qpos_without_xy"] = sim.data.qpos[2:].copy()
        obs_dict["qvel"] = sim.data.qvel[:].copy() * self.dt
        obs_dict["com_vel"] = np.array([self._get_com_velocity().copy()])
        obs_dict["torso_angle"] = np.array([self._get_torso_angle().copy()])
        obs_dict["feet_heights"] = self._get_feet_heights().copy()
        obs_dict["height"] = np.array([self._get_height()]).copy()
        obs_dict["feet_rel_positions"] = (
            self._get_feet_relative_position().copy()
        )
        obs_dict["phase_var"] = np.array(
            [(self.steps / self.hip_period) % 1]
        ).copy()
        obs_dict["muscle_length"] = self.muscle_lengths()
        obs_dict["muscle_velocity"] = self.muscle_velocities()
        obs_dict["muscle_force"] = self.muscle_forces()

        # Insert target velocity observations here
        _, y_vel = self._get_com_velocity()
        obs_dict["target_vel"] = np.array(
            [
                self.target_y_vel,
                self.target_y_vel - y_vel,
            ]
        )

        if sim.model.na > 0:
            obs_dict["act"] = sim.data.act[:].copy()

        return obs_dict


register_env_with_variants(
    id="myoLegWalk-v0-customReward",
    entry_point="deprl.custom_myosuite:WalkEnvCustomRewardV0",
    max_episode_steps=10000,
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
        "curriculum": None,  # whether to use a cuuriculum to determine target_y_vel
        "target_rot": None,  # if None then the initial root pos will be taken, otherwise provide quat
        "weighted_reward_keys": WalkEnvCustomRewardV0.DEFAULT_RWD_KEYS_AND_WEIGHTS,
        "obs_keys": WalkEnvV0.DEFAULT_OBS_KEYS + ["target_vel"],
    },
)
