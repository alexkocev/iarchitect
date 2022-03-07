import base64
import io

import IPython
import imageio
import numpy as np


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


def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)



def create_policy_eval_video(tf_env,py_env,policy, filename, num_episodes=5, fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = tf_env.reset()
            img = py_env.render()
            video.append_data(img)
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = tf_env.step(action_step.action)
                img = py_env.render()
                video.append_data(img)
    return embed_mp4(filename)


def fig_to_array(fig):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=fig.dpi)
    io_buf.seek(0)
    img_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8).reshape((int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr




