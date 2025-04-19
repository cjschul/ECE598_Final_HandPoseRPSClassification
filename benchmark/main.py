import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import os

from hand_pose_rps_dataset import hand_pose_rps_dataset


def random_axis_rotation_matrix():
    angle_x = tf.random.uniform([], -math.pi, math.pi)
    angle_y = tf.random.uniform([], -math.pi, math.pi)
    angle_z = tf.random.uniform([], -math.pi, math.pi)
    rot_x = tf.convert_to_tensor([
        [1.0, 0.0, 0.0],
        [0.0, tf.cos(angle_x), -tf.sin(angle_x)],
        [0.0, tf.sin(angle_x), tf.cos(angle_x)]
    ])
    rot_y = tf.convert_to_tensor([
        [tf.cos(angle_y), 0.0, tf.sin(angle_y)],
        [0.0, 1.0, 0.0],
        [-tf.sin(angle_y), 0.0, tf.cos(angle_y)]
    ])
    rot_z = tf.convert_to_tensor([
        [tf.cos(angle_z), -tf.sin(angle_z), 0.0],
        [tf.sin(angle_z), tf.cos(angle_z), 0.0],
        [0.0, 0.0, 1.0]
    ])
    return rot_x @ rot_y @ rot_z

def random_rotate_pointcloud(points):
    rotation_matrix = random_axis_rotation_matrix()
    return tf.linalg.matmul(points, rotation_matrix)

def duplicate_with_multiple_rotations(pose, label, num_copies=3):
    original = tf.data.Dataset.from_tensors((pose, label))
    def make_rotated(_):
        rotated = random_rotate_pointcloud(pose)
        return (rotated, label)
    rotated_ds = tf.data.Dataset.range(num_copies).map(make_rotated)
    return original.concatenate(rotated_ds)

class HandPoseAligner(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, joint_matrix):
        if len(joint_matrix.shape) == 3:
             joint_matrix = tf.expand_dims(joint_matrix, axis=0)
        if joint_matrix.shape[-1] == 1: 
             joint_matrix = tf.squeeze(joint_matrix, axis=-1)

        origin_aligned_joint_matrix = joint_matrix - joint_matrix[:, 0:1, :]

        p_wrist = origin_aligned_joint_matrix[:, 0, :]
        p_index = origin_aligned_joint_matrix[:, 5, :]
        p_pinky = origin_aligned_joint_matrix[:, 17, :]

        v_index = p_index - p_wrist
        v_pinky = p_pinky - p_wrist

        x_axis = tf.linalg.l2_normalize(v_index, axis=-1)
        z_axis = tf.linalg.cross(v_index, v_pinky)
        z_axis = tf.linalg.l2_normalize(z_axis, axis=-1)
        y_axis = tf.linalg.cross(z_axis, x_axis)

        rotation_matrix = tf.stack([x_axis, y_axis, z_axis], axis=-1)

        pose_aligned_joint_matrix = tf.matmul(origin_aligned_joint_matrix, rotation_matrix)

        pose_aligned_joint_matrix = tf.expand_dims(pose_aligned_joint_matrix, axis=-1)
        return pose_aligned_joint_matrix

class FingerDistanceIsolator(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, joint_matrix):
        if len(joint_matrix.shape) == 3: 
             joint_matrix = tf.expand_dims(joint_matrix, axis=0)
        if joint_matrix.shape[-1] == 1: 
             joint_matrix = tf.squeeze(joint_matrix, axis=-1)

        p_wrist = joint_matrix[:,0,:]
        p_thumb = joint_matrix[:,4,:]
        p_index = joint_matrix[:,8,:]
        p_middle = joint_matrix[:,12,:]
        p_ring = joint_matrix[:,16,:]
        p_pinky = joint_matrix[:,20,:]

        d_thumb = tf.linalg.norm(p_thumb - p_wrist, axis=-1)
        d_index = tf.linalg.norm(p_index - p_wrist, axis=-1)
        d_middle = tf.linalg.norm(p_middle - p_wrist, axis=-1)
        d_ring = tf.linalg.norm(p_ring - p_wrist, axis=-1)
        d_pinky = tf.linalg.norm(p_pinky - p_wrist, axis=-1)

        d_fingers = tf.stack([d_thumb, d_index, d_middle, d_ring, d_pinky], axis=-1)
        d_fingers = tf.expand_dims(d_fingers, axis=-1)
        return d_fingers


builder = hand_pose_rps_dataset.MyHandPoseRpsDataset()

try:
    train_raw = builder.as_dataset(split='train', as_supervised=True)
    test_raw = builder.as_dataset(split='test', as_supervised=True)
except Exception as e:
    print(f"error")

num_augments = 10  
train_augmented = train_raw.flat_map(lambda pose, label:
    duplicate_with_multiple_rotations(pose, label, num_augments)
)
test_augmented = test_raw.flat_map(lambda pose, label:
    duplicate_with_multiple_rotations(pose, label, num_augments)
)


def preprocess_data(pose, label):
    pose = tf.expand_dims(pose, -1)
    return pose, label

batch_size = 64
train_data = train_augmented.map(preprocess_data).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_data = test_augmented.map(preprocess_data).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_conv2d_no_align(input_shape=(21, 3, 1), num_classes=3):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LayerNormalization(axis=1)(inputs) 
    x = tf.keras.layers.Conv2D(4, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(4, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(4, 1))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Conv2D_NoAlign")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_conv2d_with_align(input_shape=(21, 3, 1), num_classes=3):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LayerNormalization(axis=1)(inputs) 
    x = HandPoseAligner()(x) 
    x = tf.keras.layers.Conv2D(4, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(4, 1))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Conv2D_WithAlign")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_mlp_finger_dist(input_shape=(21, 3, 1), num_classes=3):
    inputs = tf.keras.Input(shape=input_shape)
    x = FingerDistanceIsolator()(inputs) 
    x = tf.keras.layers.LayerNormalization(axis=1)(x) 
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MLP_FingerDist")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

models_to_benchmark = {
    "Conv2D (No Alignment)": build_conv2d_no_align(),
    "Conv2D (With Alignment)": build_conv2d_with_align(),
    "MLP (Finger Distance)": build_mlp_finger_dist()
}

results = {}
epochs = 10 
num_timing_runs = 100 

sample_batch, _ = next(iter(test_data))

for name, model in models_to_benchmark.items():
    model.summary()

    start_train_time = time.time()
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=test_data,
        verbose=1 
    )
    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    loss, accuracy = model.evaluate(test_data, verbose=0)

    _ = model(sample_batch, training=False)

    start_inference_time = time.time()
    for _ in range(num_timing_runs):
        _ = model(sample_batch, training=False)
    end_inference_time = time.time()

    avg_inference_time_batch = (end_inference_time - start_inference_time) / num_timing_runs
    avg_inference_time_sample = avg_inference_time_batch / sample_batch.shape[0]

    results[name] = {
        "loss": loss,
        "accuracy": accuracy,
        "training_time_s": training_time,
        "avg_inference_time_batch_ms": avg_inference_time_batch * 1000,
        "avg_inference_time_sample_ms": avg_inference_time_sample * 1000,
        "history": history.history 
    }



print("\n\n--- Benchmark Summary ---")
for name, metrics in results.items():
    print(f"Model: {name}")
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test Loss: {metrics['loss']:.4f}")
    print(f"  Training Time: {metrics['training_time_s']:.2f} s")
    print(f"  Avg Inference Time (Batch): {metrics['avg_inference_time_batch_ms']:.4f} ms")
    print(f"  Avg Inference Time (Sample): {metrics['avg_inference_time_sample_ms']:.4f} ms")
    print("-" * 20)

num_models = len(results)
fig, axes = plt.subplots(2, num_models, figsize=(5 * num_models, 10), squeeze=False)
fig.suptitle('Model Training History')

model_index = 0
for name, metrics in results.items():
    history = metrics['history']

    axes[0, model_index].plot(history['loss'], label='Train Loss')
    axes[0, model_index].plot(history['val_loss'], label='Validation Loss')
    axes[0, model_index].set_title(f'{name}\nLoss')
    axes[0, model_index].set_ylabel('Loss')
    axes[0, model_index].set_xlabel('Epoch')
    axes[0, model_index].legend()

    axes[1, model_index].plot(history['accuracy'], label='Train Accuracy')
    axes[1, model_index].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[1, model_index].set_title(f'{name}\nAccuracy')
    axes[1, model_index].set_ylabel('Accuracy')
    axes[1, model_index].set_xlabel('Epoch')
    axes[1, model_index].legend()

    model_index += 1

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.show()