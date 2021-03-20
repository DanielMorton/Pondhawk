import tensorflow as tf

from layers import CoarseSamples, FineSamples, RGB, TrigEmbed

COARSE_SAMPLES = 64
FINE_SAMPLES = 128
CHANNELS = 256
LAYERS = 8
COORD_EMBED = 10
DIR_EMBED = 4


def make_point_model(channels, layers, coord_dim, dir_dim, name):
    """Constructs the MLP for computing RGBalpha for individual points."""
    coordinate_input = tf.keras.layers.Input(shape=coord_dim, name='coordinate')
    direction_input = tf.keras.layers.Input(shape=dir_dim, name='viewdir')
    for n in range(layers):
        if n == 0:
            hidden_output = tf.keras.layers.Dense(channels,
                                                  activation='relu',
                                                  name=f'dense_{n}')(coordinate_input)
        elif (n - 1) == (layers // 2):
            hidden_output = tf.keras.layers.Concatenate(name='concat_0')([coordinate_input, hidden_output])
            hidden_output = tf.keras.layers.Dense(channels,
                                                  activation='relu',
                                                  name=f'dense_{n}')(hidden_output)
        else:
            hidden_output = tf.keras.layers.Dense(channels,
                                                  activation='relu',
                                                  name=f'dense_{n}')(hidden_output)
    density_output = tf.keras.layers.Dense(1, activation=None,
                                           name='density_out')(hidden_output)
    hidden_output = tf.keras.layers.Dense(channels,
                                          activation=None,
                                          name=f'dense_{layers}')(hidden_output)
    color_output = tf.keras.layers.Concatenate(name='concat_1')([hidden_output, direction_input])
    color_output = tf.keras.layers.Dense(channels // 2,
                                         activation='relu',
                                         name='dense_half')(color_output)
    color_output = tf.keras.layers.Dense(3, activation='sigmoid',
                                         name='color_out')(color_output)
    model_output = {'density_output': density_output,
                    'color_output': color_output}
    point_model = tf.keras.models.Model(inputs=[coordinate_input, direction_input],
                                        outputs=model_output,
                                        name=name)
    return point_model


def make_coarse_model(coarse_inputs,
                      coordinate_embedder,
                      direction_embedder,
                      args):
    """ Constructs the coarse component of the model."""
    coarse_points, coarse_veiws, viewdirs, ndc_origin_input, ndc_direction_input, = CoarseSamples(
        sample_size=COARSE_SAMPLES,
        H=args.heigh, W=args.width, focal=arg.focal,
        name='coarse_sample')(coarse_inputs)
    coarse_points = tf.keras.layers.Concatenate(axis=0)(coarse_points)
    coarse_points = coordinate_embedder(coarse_points)
    coarse_veiws = tf.keras.layers.Concatenate(axis=0)(coarse_veiws)
    coarse_veiws = direction_embedder(coarse_veiws)

    coarse_model = make_point_model(CHANNELS, LAYERS,
                                    coord_dim=coarse_points.shape[1],
                                    dir_dim=coarse_veiws.shape[1],
                                    name='coarse_model')

    coarse_outputs = coarse_model([coarse_points, coarse_veiws])
    coarse_density = coarse_outputs['density_output']
    coarse_density = tf.transpose(tf.reshape(coarse_density, (COARSE_SAMPLES, 4096)), (1, 0))
    coarse_color = coarse_outputs['color_output']
    coarse_color = tf.transpose(tf.reshape(coarse_color, (COARSE_SAMPLES, 4096, 3)), (1, 0, 2))

    coarse_rgb_inputs = {'density': coarse_density,
                         'color': coarse_color,
                         'z_vals': coarse_inputs['z_vals'],
                         'direction': ndc_direction_input}
    coarse_rgb_output = RGB(name='coarse_rgb', weights_out=True)(coarse_rgb_inputs)
    return coarse_rgb_output, viewdirs, ndc_origin_input, ndc_direction_input


def make_fine_model(fine_inputs,
                    coordinate_embedder,
                    direction_embedder):
    """Constructs the fine component of the model"""
    fine_points, fine_views, z_vals_fine = FineSamples(sample_size=FINE_SAMPLES,
                                                       name='fine_sample')(fine_inputs)

    fine_points = tf.keras.layers.Concatenate(axis=0)(fine_points)
    fine_points = coordinate_embedder(fine_points)
    fine_views = tf.keras.layers.Concatenate(axis=0)(fine_views)
    fine_views = direction_embedder(fine_views)

    fine_model = make_point_model(CHANNELS, LAYERS,
                                  coord_dim=fine_points.shape[1],
                                  dir_dim=fine_views.shape[1],
                                  name='fine_model')

    fine_outputs = fine_model([fine_points, fine_views])
    fine_density = fine_outputs['density_output']
    fine_density = tf.transpose(tf.reshape(fine_density, (COARSE_SAMPLES + FINE_SAMPLES, 4096)), (1, 0))
    fine_color = fine_outputs['color_output']
    fine_color = tf.transpose(tf.reshape(fine_color, (COARSE_SAMPLES + FINE_SAMPLES, 4096, 3)), (1, 0, 2))

    fine_rgb_inputs = {'density': fine_density,
                       'color': fine_color,
                       'z_vals': z_vals_fine,
                       'direction': fine_inputs['direction_inputs']}

    fine_rgb_output = RGB(name='fine_rgb')(fine_rgb_inputs)
    return fine_rgb_output


def make_model():
    """Constructs the full model"""
    origin_input = tf.keras.layers.Input(shape=3, batch_size=4096, name='origin_input')
    direction_input = tf.keras.layers.Input(shape=3, batch_size=4096, name='direction')
    z_vals = tf.keras.layers.Input(shape=COARSE_SAMPLES, batch_size=4096, name='z_vals')
    u = tf.keras.layers.Input(shape=FINE_SAMPLES, batch_size=4096, name='u')

    coarse_inputs = {'origin_input': origin_input,
                     'direction_input': direction_input,
                     'z_vals': z_vals}

    coordinate_embedder = TrigEmbed(embed_dim=COORD_EMBED, name='coordinate_embeder')
    direction_embedder = TrigEmbed(embed_dim=DIR_EMBED, name='direction_embedder')
    coarse_rgb_output, viewdirs, ndc_origin_input, ndc_direction_input = make_coarse_model(
        coarse_inputs,
        coordinate_embedder,
        direction_embedder)

    fine_inputs = {'origin_input': ndc_origin_input,
                   'direction_input': ndc_direction_input,
                   'z_vals': z_vals,
                   'viewdirs': viewdirs,
                   'weights': coarse_rgb_output['weights'],
                   'u': u}

    fine_rgb_output = make_fine_model(fine_inputs,
                                      coordinate_embedder,
                                      direction_embedder)

    model_inputs = {'origin_input': origin_input,
                    'direction_input': direction_input,
                    'z_vals': z_vals,
                    'u': u}
    mode_outputs = {'coarse_output': coarse_rgb_output['rgb'],
                    'fine_output': fine_rgb_output['rgb']}

    full_model = tf.keras.Model(inputs=model_inputs,
                                outputs=mode_outputs)
    return full_model

