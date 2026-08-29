import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch


class FlockingCallbacks(DefaultCallbacks):

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        info = episode.last_info_for()
        if info is None:
            return

        sp = info.get("spatial_entropy")
        vl = info.get("velocity_entropy")
        if sp is not None:
            episode.custom_metrics["final_spatial_entropy"] = float(sp)
        if vl is not None:
            episode.custom_metrics["final_velocity_entropy"] = float(vl)

        cr = info.get("conn_ratio")
        if cr is not None:
            episode.custom_metrics["final_conn_ratio"] = float(cr)

        # flocking_success reads "the episode terminated before the cap", which
        # only carries information when early termination is enabled. Under a
        # fixed episode length every episode runs the full horizon, so the
        # metric is a constant and only clutters the progress table.
        max_steps, fixed_length = 1000, False
        try:
            env = base_env.get_sub_environments()[0]
            max_steps = env.config.env.max_time_steps
            fixed_length = bool(env.config.env.use_fixed_episode_length)
        except Exception:
            pass
        if not fixed_length:
            episode.custom_metrics["flocking_success"] = float(
                episode.length < max_steps)

    def on_postprocess_trajectory(self, *, worker, episode, agent_id,
                                  policy_id, policies, postprocessed_batch,
                                  original_batches, **kwargs):
        infos = postprocessed_batch[SampleBatch.INFOS]
        first_par = infos[0].get("per_agent_rewards") if len(infos) > 0 else None
        if first_par is not None:
            N = len(first_par)
            per_agent_rewards = np.array(
                [info.get("per_agent_rewards", np.zeros(N, dtype=np.float32))
                 for info in infos],
                dtype=np.float32,
            )
            postprocessed_batch["per_agent_rewards"] = per_agent_rewards


class C2Callbacks(FlockingCallbacks):
    """FlockingCallbacks + C2-criterion training support (study acs-c2-train).

    Adds per-episode C2 metrics (c2_success, t_conv, J from raw costs, final
    swarm stats) and drives the dist_aux coefficient anneal on the learner's
    model and its GPU towers (custom_loss runs on the towers in RLlib 2.1.0,
    so mutating policy.model alone would not reach the training graph).
    """

    def on_episode_step(self, *, worker, base_env, policies=None, episode, **kwargs):
        info = episode.last_info_for()
        if info is not None and info.get("original_reward") is not None:
            episode.user_data["j_sum"] = (
                episode.user_data.get("j_sum", 0.0) + float(info["original_reward"]))

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        super().on_episode_end(worker=worker, base_env=base_env,
                               policies=policies, episode=episode, **kwargs)
        info = episode.last_info_for()
        if info is None:
            return
        if "c2_success" in info:
            success = bool(info["c2_success"])
            episode.custom_metrics["c2_success"] = float(success)
            if success:
                episode.custom_metrics["t_conv"] = float(info.get("t_conv", episode.length))
                if "j_sum" in episode.user_data:
                    # J = cumulative per-agent raw cost to convergence
                    episode.custom_metrics["J_success"] = -float(episode.user_data["j_sum"])
        if "j_sum" in episode.user_data:
            episode.custom_metrics["J_episode"] = -float(episode.user_data["j_sum"])
        for k in ("c2_phi", "c2_f_largest", "c2_n_comp"):
            if info.get(k) is not None:
                episode.custom_metrics[f"final_{k}"] = float(info[k])

    def on_learn_on_batch(self, *, policy, train_batch, result, **kwargs):
        model = getattr(policy, "model", None)
        sched = getattr(model, "dist_aux_schedule", None)
        if model is None or not sched:
            return
        xs = [float(p[0]) for p in sched]
        ys = [float(p[1]) for p in sched]
        coef = float(np.interp(float(policy.global_timestep), xs, ys))
        towers = [m for m in getattr(policy, "model_gpu_towers", []) or []]
        for m in [model] + [t for t in towers if t is not model]:
            m.dist_aux_coef_current = coef
