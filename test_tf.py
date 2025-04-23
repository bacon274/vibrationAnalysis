import torch
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from models import Autoencoder
from dataset import create_data_loaders
from torch.nn.modules.loss import MSELoss
from sklearn.metrics import mean_squared_error

def load_pytorch_model(model_path, input_shape):
    """Load and prepare PyTorch model"""
    # Add MSELoss to safe globals
    torch.serialization.add_safe_globals([MSELoss])
    
    # Load checkpoint with weights_only=False since we trust the source
    checkpoint = torch.load(model_path, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Create and load model
    model = Autoencoder(
        sequence_length=input_shape[1],
        n_features=input_shape[2],
        hidden_dim=64,
        latent_dim=12,
        num_layers=4
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_tensorflow_model(model_path):
    """Load TensorFlow SavedModel"""
    model = tf.saved_model.load(str(model_path))
    # Get the expected input shape
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    input_shape = concrete_func.inputs[0].shape
    print(f"TensorFlow model expects input shape: {input_shape}")
    return model, input_shape

def load_tflite_model(model_path):
    """Load TensorFlow Lite model"""
    # Convert Path to string
    model_path_str = str(model_path)
    
    # Create interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path_str)
    interpreter.allocate_tensors()
    
    # Get input details
    input_details = interpreter.get_input_details()
    input_shape = tuple(input_details[0]['shape'])  # Convert to tuple for comparison
    print(f"TensorFlow Lite model expects input shape: {input_shape}")
    
    return interpreter, input_shape

def run_inference_pytorch(model, input_data):
    """Run inference with PyTorch model"""
    with torch.no_grad():
        output = model(input_data)
    return output.numpy()

def run_inference_tensorflow(model, input_data, expected_shape):
    """Run inference with TensorFlow model"""
    # Reshape input to match expected shape
    input_np = input_data.numpy()
    if tuple(input_np.shape) != tuple(expected_shape):
        # Reshape to (batch_size, n_features, sequence_length)
        input_np = np.transpose(input_np, (0, 2, 1))
    
    # Convert to TensorFlow format
    input_tensor = tf.convert_to_tensor(input_np)
    
    # Run inference
    output = model(input_tensor)
    return output.numpy()

def run_inference_tflite(interpreter, input_data, expected_shape):
    """Run inference with TensorFlow Lite model"""
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Reshape input to match expected shape
    input_np = input_data.numpy()
    if tuple(input_np.shape) != expected_shape:
        # Reshape to (batch_size, n_features, sequence_length)
        input_np = np.transpose(input_np, (0, 2, 1))
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_np)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

def calculate_mse(input_data, output_data, model_name):
    """Calculate MSE between input and output"""
    # Flatten the arrays for MSE calculation
    input_flat = input_data.flatten()
    output_flat = output_data.flatten()
    mse = mean_squared_error(input_flat, output_flat)
    print(f"{model_name} MSE: {mse:.6f}")
    return mse

def plot_comparison(input_data, pytorch_output, tf_output, tflite_output, save_path='model_comparison.png'):
    """Plot comparison of model outputs"""
    plt.figure(figsize=(15, 10))
    
    # Plot input
    plt.subplot(2, 1, 1)
    plt.plot(input_data[0, :, 0], 'k-', label='Input', alpha=0.7)
    plt.title('Input Signal')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Plot outputs
    plt.subplot(2, 1, 2)
    plt.plot(pytorch_output[0, :, 0], 'b-', label='PyTorch', alpha=0.7)
    plt.plot(tf_output[0, :, 0], 'g-', label='TensorFlow', alpha=0.7)
    plt.plot(tflite_output[0, :, 0], 'r-', label='TensorFlow Lite', alpha=0.7)
    plt.title('Model Outputs Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Define paths
    checkpoint_dir = Path('checkpoints')
    pytorch_path = checkpoint_dir / 'autoencoder_checkpoint.pth'
    tf_path = checkpoint_dir / 'autoencoder_checkpoint_tf'
    tflite_path = checkpoint_dir / 'autoencoder_checkpoint.tflite'
    
    # Create data loader
    train_loader, _ = create_data_loaders(
        csv_path='./data/combined/normal_500Hz.csv',
        selected_columns=['underhang_bearing_axial'],
        sequence_length=128,
        batch_size=1,
        train_split=0.8,
        domain='time',
        normalise=False,
        random_sampling=True
    )
    
    # Get a sample batch
    sample_batch = next(iter(train_loader))
    input_shape = sample_batch.shape
    
    # Load models
    print("Loading models...")
    pytorch_model = load_pytorch_model(pytorch_path, input_shape)
    tf_model, tf_input_shape = load_tensorflow_model(tf_path)
    tflite_interpreter, tflite_input_shape = load_tflite_model(tflite_path)
    
    # Run inference
    print("\nRunning inference...")
    pytorch_output = run_inference_pytorch(pytorch_model, sample_batch)
    tf_output = run_inference_tensorflow(tf_model, sample_batch, tf_input_shape)
    tflite_output = run_inference_tflite(tflite_interpreter, sample_batch, tflite_input_shape)
    
    # Calculate and print MSEs
    print("\nModel Performance (MSE):")
    print("-" * 30)
    pytorch_mse = calculate_mse(sample_batch.numpy(), pytorch_output, "PyTorch")
    tf_mse = calculate_mse(sample_batch.numpy(), tf_output, "TensorFlow")
    tflite_mse = calculate_mse(sample_batch.numpy(), tflite_output, "TensorFlow Lite")
    print("-" * 30)
    
    # Plot comparison
    print("\nPlotting results...")
    plot_comparison(sample_batch, pytorch_output, tf_output, tflite_output)
    print("Comparison plot saved as 'model_comparison.png'")

if __name__ == "__main__":
    main()
