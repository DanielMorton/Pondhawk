import tensorflow as tf


class CoarseSamples(tf.keras.layers.Layer):
    '''Converts ray data (origin and direction) to evenly spaced point samples for
    coarse model input.
    Args:
      sample_size: Number of points to sample
      ndc: Whether to convert ray origin and direction to NDC coordiates
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      name: Layer name
    '''
    def __init__(self, sample_size, ndc=True, H=None, W=None, focal=None,
                 name=None):
        super(CoarseSamples, self).__init__(name=name)
        self.sample_size = sample_size
        t_vals = tf.linspace(0., 1., sample_size)
        self.ndc = ndc
        self.H, self.W, self.focal = H, W, focal
        self.near = tf.cast(1., tf.float32)

    def ndc_rays(self, rays_o, rays_d):
        '''Converts ray orign and direction to NDC coordiantes'''
        t = -(self.near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d

        # Projection
        o0 = -1 / (self.W / (2 * self.focal)) * rays_o[..., 0] / rays_o[..., 2]
        o1 = -1 / (self.H / (2 * self.focal)) * rays_o[..., 1] / rays_o[..., 2]
        o2 = 1 + 2 * self.near / rays_o[..., 2]

        d0 = -1 / (self.W / (2 * self.focal)) * \
             (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
        d1 = -1 / (self.H / (2 * self.focal)) * \
             (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
        d2 = -2 * self.near / rays_o[..., 2]

        rays_o = tf.stack([o0, o1, o2], -1)
        rays_d = tf.stack([d0, d1, d2], -1)

        return rays_o, rays_d

    def call(self, inputs):
        '''Calls the coarse sample layer'''
        rays_o, rays_d, z_vals = inputs['origin_input'], inputs['direction_input'], inputs['z_vals']
        viewdirs = rays_d
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)
        if self.ndc:
            rays_o, rays_d = self.ndc_rays(rays_o, rays_d)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts_list = [pts[:, i, :] for i in range(self.sample_size)]
        view_list = [viewdirs for i in range(self.sample_size)]
        return pts_list, view_list, viewdirs, rays_o, rays_d,
