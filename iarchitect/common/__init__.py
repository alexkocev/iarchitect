import io

import numpy as np
from PIL import Image

from iarchitect.render import create_list_video, create_list_gif


def show_policy_behaviour(environment,policy,max_iter,one_episode_only=True):
    time_step = environment.current_time_step()
    iter_ = 0.0
    results = [(None,None,time_step.observation.numpy())]
    while (not time_step.is_last() or not one_episode_only) and iter_<=max_iter:
        iter_+=1
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        results.append((action_step.action.numpy(),time_step.reward.numpy(),time_step.observation.numpy()))
    return results

def common_create_policy_eval(tf_env,py_env,
                              policy,
                              num_episodes=5,
                              each_n_action=1):
    images = []
    for _ in range(num_episodes):
        time_step = tf_env.reset()
        img = py_env.render(mode="rgb_array")
        images.append(Image.fromarray(img))
        iaction = 0
        plotted = True
        while not time_step.is_last():
            plotted = False
            action_step = policy.action(time_step)
            time_step = tf_env.step(action_step.action)
            if iaction % each_n_action ==0:
                img = py_env.render(mode="rgb_array")
                images.append(Image.fromarray(img))
                plotted = True
            iaction += 1
        if not plotted:
            img = py_env.render(mode="rgb_array")
            images.append(Image.fromarray(img))
    return images

def create_policy_eval_gif(tf_env,py_env,
                           policy,
                           filename,
                           num_episodes=5,
                           fps=30,each_n_action=1):
    images = common_create_policy_eval(tf_env,py_env,policy,num_episodes=num_episodes,each_n_action=each_n_action)
    return create_list_gif(images,filename,fps=fps)

def create_policy_eval_video(tf_env,py_env,
                             policy,
                             filename,
                             num_episodes=5,
                             fps=30,each_n_action=1,
                             embed=True):
    images = common_create_policy_eval(tf_env,py_env,policy,num_episodes=num_episodes,each_n_action=each_n_action)
    return create_list_video(images,filename,embed=embed,fps=fps)


def fig_to_array(fig):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=fig.dpi)
    io_buf.seek(0)
    img_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8).reshape((int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr




