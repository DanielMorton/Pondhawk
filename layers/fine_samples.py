import tensorflow as tf


class FineSamples(tf.keras.layers.Layer):
    '''Converts ray data (origin and direction) to samples for
        fine model input.
        Args:
          sample_size: Number of points to sample
          name: Layer name
        '''
    def __init__(self, sample_size, name=None):
        super(FineSamples, self).__init__(name=name)
        self.sample_size = sample_size

    def sample_pdf(self, u, bins, weights):
        """ Creates samples along ray.
        Args:
            u: Distribution of sample probabilities along rays
            bins: Number of samples to compute
            weights: Unnormalized pdf"""
        weights += 1e-5  # prevent nans
        pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
        print(f"pdf {pdf}")
        cdf = tf.cumsum(pdf, -1)
        cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)
        inds = tf.searchsorted(cdf, u, side='right')
        print(f"inds {inds}")
        below = tf.maximum(0, inds - 1)
        above = tf.minimum(tf.shape(cdf)[-1] - 1, inds)
        inds_g = tf.stack([below, above], -1)
        cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(tf.shape(inds_g)) - 2)
        bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(tf.shape(inds_g)) - 2)

        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples

    def call(self, inputs):
        rays_o, rays_d, z_vals = inputs['origin_input'], inputs['direction_input'], inputs['z_vals']
        viewdirs, weights, u = inputs['viewdirs'], inputs['weights'], inputs['u']
        input_shape = z_vals.shape[-1]

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = self.sample_pdf(u, z_vals_mid, weights[..., 1:-1])
        z_vals = tf.concat([z_vals, z_samples], -1)
        z_vals = tf.sort(z_vals, -1)

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts_list = [pts[:, i, :] for i in range(self.sample_size + input_shape)]
        view_list = [viewdirs for i in range(self.sample_size + input_shape)]
        return pts_list, view_list, z_vals
