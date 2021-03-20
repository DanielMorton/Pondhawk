import tensorflow as tf


class RGB(tf.keras.layers.Layer):
    """Computes the total RGB value for each ray from sample points.
    Args:
        raw_noise_std: Amount of noise to add to alpha estimates. Makes predictions more robust.
        weights_out: Return the unnormalized weights if needed later.
        name: Layer Name.
    """
    def __init__(self, raw_noise_std=0,
                 weights_out=False,
                 name=None):
        super(RGB, self).__init__(name=name)
        self.raw_noise_std = raw_noise_std
        self.weights_out = weights_out

    def call(self, inputs):
        density, color, z_vals, rays_d = inputs['density'], inputs['color'], inputs['z_vals'], inputs['direction']
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = tf.concat(
            [dists, tf.broadcast_to([1e10], tf.shape(dists[..., :1]))],
            axis=-1)
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        noise = tf.random.normal(tf.shape(density)) * self.raw_noise_std
        density += noise
        alpha = 1 - tf.exp(-tf.nn.relu(density) * dists)
        weights = alpha * tf.math.cumprod(1. - alpha + 1e-10, axis=-1, exclusive=True)

        # Computed weighted color of each sample along each ray.
        rgb_map = {'rgb': tf.reduce_sum(weights[..., None] * color, axis=-2)}  # [N_rays, 3]
        if self.weights_out:
            rgb_map['weights'] = weights
        return rgb_map
