"""
Medical Triage Reinforcement Learning Environment
==================================================
A custom Gymnasium environment simulating an emergency medical triage unit.
The agent acts as a triage coordinator deciding how to allocate limited
hospital resources (beds, staff, equipment) across multiple simultaneous
patients of varying severity levels (1-5), while patients deteriorate
over time if left untreated.

Author : Christian Ishimwe
Course : Machine Learning & Robotics - ALU
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
MAX_PATIENTS        = 8
NUM_SEVERITY_LEVELS = 5          # ESI 1 (critical) -> 5 (non-urgent)
MAX_BEDS            = 6
MAX_STAFF           = 4
MAX_EQUIPMENT       = 3
MAX_STEPS           = 200
DETERIORATION_RATE  = 0.05

REWARD_TREAT_CRITICAL  =  10.0
REWARD_TREAT_MODERATE  =   5.0
REWARD_TREAT_MINOR     =   2.0
PENALTY_PATIENT_DEATH  = -20.0
PENALTY_WRONG_PRIORITY =  -3.0
PENALTY_RESOURCE_WASTE =  -1.0
REWARD_DISCHARGE       =   3.0


class MedicalTriageEnv(gym.Env):
    """
    Observation: flattened patient matrix (MAX_PATIENTS x 5 features) + 3 resource scalars
    Each patient row: [severity/5, wait/MAX_STEPS, is_treated, is_alive, deterioration/10]
    Resources: [beds/MAX_BEDS, staff/MAX_STAFF, equip/MAX_EQUIPMENT]
    Total obs dim = 8*5 + 3 = 43

    Actions: Discrete(9)
        0         -> hold (do nothing)
        1..8      -> treat patient at index (action-1)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        obs_size = MAX_PATIENTS * 5 + 3
        self.observation_space = spaces.Box(
            low=np.zeros(obs_size, dtype=np.float32),
            high=np.ones(obs_size, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(MAX_PATIENTS + 1)

        self.patients: np.ndarray = None
        self.beds_available: int = MAX_BEDS
        self.staff_available: int = MAX_STAFF
        self.equipment_available: int = MAX_EQUIPMENT
        self.step_count: int = 0
        self.total_deaths: int = 0
        self.total_treated: int = 0
        self._renderer = None

    # ── reset ───────────────────────────────────
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        n_patients = self.np_random.integers(3, MAX_PATIENTS + 1)
        self.patients = np.zeros((MAX_PATIENTS, 5), dtype=np.float32)
        for i in range(n_patients):
            self.patients[i] = [
                float(self.np_random.integers(1, 6)),
                0.0, 0.0, 1.0, 0.0,
            ]

        self.beds_available      = MAX_BEDS
        self.staff_available     = MAX_STAFF
        self.equipment_available = MAX_EQUIPMENT
        self.step_count          = 0
        self.total_deaths        = 0
        self.total_treated       = 0

        return self._get_obs(), self._get_info()

    # ── step ────────────────────────────────────
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1
        reward = 0.0

        # 1. Execute action
        if action == 0:
            critical_untreated = np.sum(
                (self.patients[:, 0] == 1) &
                (self.patients[:, 2] == 0) &
                (self.patients[:, 3] == 1)
            )
            reward += PENALTY_RESOURCE_WASTE * float(critical_untreated)
        else:
            idx = action - 1
            p = self.patients[idx]
            if p[3] == 0:
                reward += PENALTY_WRONG_PRIORITY
            elif p[2] == 1:
                reward += PENALTY_RESOURCE_WASTE
            elif self._resources_available(p[0]):
                self.patients[idx, 2] = 1.0
                self._consume_resources(p[0])
                self.total_treated += 1
                sev = int(p[0])
                if sev == 1:
                    reward += REWARD_TREAT_CRITICAL
                elif sev in (2, 3):
                    reward += REWARD_TREAT_MODERATE
                else:
                    reward += REWARD_TREAT_MINOR
                if self._is_highest_priority(idx):
                    reward += 2.0
            else:
                reward += PENALTY_WRONG_PRIORITY

        # 2. Deterioration and wait time
        for i in range(MAX_PATIENTS):
            p = self.patients[i]
            if p[3] == 0 or p[0] == 0:
                continue
            p[1] = min(p[1] + 1, MAX_STEPS)
            if p[2] == 0:
                det_prob = DETERIORATION_RATE * (6.0 - p[0]) / 5.0
                if self.np_random.random() < det_prob:
                    p[4] += 1
                    if p[4] >= 3 and p[0] == 1:
                        p[3] = 0.0
                        self.total_deaths += 1
                        reward += PENALTY_PATIENT_DEATH
                    elif p[0] > 1 and p[4] >= 5:
                        p[0] = max(1.0, p[0] - 1.0)

        # 3. Discharge treated patients
        for i in range(MAX_PATIENTS):
            p = self.patients[i]
            if p[2] == 1 and p[3] == 1 and p[1] > 10:
                reward += REWARD_DISCHARGE
                self.patients[i] = np.zeros(5)
                self._release_resources()

        # 4. Admit new patient (15% chance)
        if self.np_random.random() < 0.15:
            self._admit_new_patient()

        # 5. Termination conditions
        all_empty = np.all((self.patients[:, 3] == 0) | (self.patients[:, 0] == 0))
        terminated = bool(all_empty) or self.total_deaths >= 3
        truncated  = self.step_count >= MAX_STEPS

        if self.render_mode == "human" and self._renderer:
            self._renderer.update(self)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ── helpers ─────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        pat = self.patients.copy()
        pat[:, 0] /= NUM_SEVERITY_LEVELS
        pat[:, 1] /= MAX_STEPS
        pat[:, 4] /= 10.0
        resources = np.array([
            self.beds_available      / MAX_BEDS,
            self.staff_available     / MAX_STAFF,
            self.equipment_available / MAX_EQUIPMENT,
        ], dtype=np.float32)
        return np.concatenate([pat.flatten(), resources]).astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        alive = int(np.sum((self.patients[:, 3] == 1) & (self.patients[:, 0] > 0)))
        return {
            "step": self.step_count,
            "total_deaths": self.total_deaths,
            "total_treated": self.total_treated,
            "patients_alive": alive,
            "beds_available": self.beds_available,
            "staff_available": self.staff_available,
            "equip_available": self.equipment_available,
        }

    def _resources_available(self, severity: float) -> bool:
        if severity <= 2:
            return (self.beds_available >= 1 and
                    self.staff_available >= 2 and
                    self.equipment_available >= 1)
        elif severity == 3:
            return self.beds_available >= 1 and self.staff_available >= 1
        else:
            return self.beds_available >= 1

    def _consume_resources(self, severity: float):
        if severity <= 2:
            self.beds_available      = max(0, self.beds_available - 1)
            self.staff_available     = max(0, self.staff_available - 2)
            self.equipment_available = max(0, self.equipment_available - 1)
        elif severity == 3:
            self.beds_available  = max(0, self.beds_available - 1)
            self.staff_available = max(0, self.staff_available - 1)
        else:
            self.beds_available = max(0, self.beds_available - 1)

    def _release_resources(self):
        self.beds_available      = min(MAX_BEDS,      self.beds_available + 1)
        self.staff_available     = min(MAX_STAFF,     self.staff_available + 1)
        self.equipment_available = min(MAX_EQUIPMENT, self.equipment_available + 1)

    def _admit_new_patient(self):
        empty = np.where(self.patients[:, 0] == 0)[0]
        if len(empty) == 0:
            return
        self.patients[empty[0]] = [
            float(self.np_random.integers(1, 6)),
            0.0, 0.0, 1.0, 0.0,
        ]

    def _is_highest_priority(self, idx: int) -> bool:
        my_sev = self.patients[idx, 0]
        for i in range(MAX_PATIENTS):
            if i == idx:
                continue
            p = self.patients[i]
            if p[3] == 1 and p[2] == 0 and p[0] > 0 and p[0] < my_sev:
                return False
        return True

    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                from environment.rendering import TriageRenderer
                self._renderer = TriageRenderer()
            self._renderer.update(self)

    def close(self):
        if self._renderer:
            self._renderer.close()
            self._renderer = None


# ── quick sanity check ──────────────────────
if __name__ == "__main__":
    env = MedicalTriageEnv()
    obs, info = env.reset(seed=42)
    print("Obs shape  :", obs.shape)
    print("Action space:", env.action_space)
    print("Info       :", info)
    total_r = 0.0
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total_r += r
        if term or trunc:
            break
    print(f"Random rollout reward: {total_r:.2f}")
    env.close()
