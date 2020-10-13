import tensorflow as tf 

num_parallel = tf.data.experimental.AUTOTUNE

def decode_image(image, image_size, file_type):
    if (file_type == 'jpg'):
        image = tf.image.decode_jpeg(image, channels=3)
    else:
        image = tf.image.decode_png(image, channels=3)

    image = tf.cast(image, tf.float32) / 127.5 - 1
    image = tf.reshape(image, image_size)
    return image

def read_tfrecord(example):
    tfrecord_format = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image_raw'], 
                         [example['height'], example['width'], example['depth']],
                         str(example['image_name']).split('.')[-1])
    
    return image, example['image_name']

def load_dataset(filenames, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
        
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=num_parallel)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=num_parallel)
    return dataset

def get_dataset(filenames, augment=None, repeat=True, shuffle=True, batch_size=1):
    dataset = load_dataset(filenames)
    
    if augment:
        dataset = dataset.map(augment, num_parallel_calls=num_parallel)
    if repeat:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(1024)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(num_parallel)

    return dataset