import torch
import onnx
import onnx2tf
import tensorflow as tf
from pathlib import Path
from torch.nn.modules.loss import MSELoss
from models import Autoencoder  # Import your model class

def convert_pytorch_to_onnx(pytorch_model_path, onnx_model_path, input_shape):
    """
    Convert a PyTorch model to ONNX format.
    
    Args:
        pytorch_model_path (str): Path to the PyTorch model file
        onnx_model_path (str): Path to save the ONNX model
        input_shape (tuple): Shape of the input tensor (batch_size, sequence_length, n_features)
    """
    # Add MSELoss to safe globals
    torch.serialization.add_safe_globals([MSELoss])
    
    # Load the checkpoint
    checkpoint = torch.load(pytorch_model_path, weights_only=False)
    
    # Extract model state dict from checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Create model instance and load state dict
    model = Autoencoder(
        sequence_length=input_shape[1],  # Use sequence_length from input_shape
        n_features=input_shape[2],       # Use n_features from input_shape
        hidden_dim=64,
        latent_dim=12,
        num_layers=4
    )
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode
    
    # Create dummy input with correct shape
    dummy_input = torch.randn(input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify the ONNX model
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved to {onnx_model_path}")

def convert_onnx_to_tensorflow(onnx_model_path, tf_model_path):
    """
    Convert an ONNX model to TensorFlow format using onnx2tf.
    
    Args:
        onnx_model_path (str): Path to the ONNX model file
        tf_model_path (str): Path to save the TensorFlow model
    """
    # Convert ONNX to TensorFlow using onnx2tf
    onnx2tf.convert(
        input_onnx_file_path=onnx_model_path,
        output_folder_path=tf_model_path,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True
    )
    print(f"TensorFlow model saved to {tf_model_path}")

def convert_tf_to_tflite(tf_model_path, tflite_model_path, input_shape):
    """
    Convert a TensorFlow model to TensorFlow Lite format.
    
    Args:
        tf_model_path (str): Path to the TensorFlow model
        tflite_model_path (str): Path to save the TensorFlow Lite model
        input_shape (tuple): Shape of the input tensor (batch_size, sequence_length, n_features)
    """
    # Load the TensorFlow model
    model = tf.saved_model.load(tf_model_path)
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    
    # Get the expected input shape from the model
    expected_shape = concrete_func.inputs[0].shape
    print(f"Model expects input shape: {expected_shape}")
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # Set optimization flags
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TensorFlow Lite model saved to {tflite_model_path}")

def main():
    # Define paths
    pytorch_model_path = "autoencoder_checkpoint.pth"
    onnx_model_path = "autoencoder_checkpoint.onnx"
    tf_model_path = "autoencoder_checkpoint_tf"
    tflite_model_path = "autoencoder_checkpoint.tflite"
    
    # Define input shape (batch_size, sequence_length, n_features)
    input_shape = (1, 128, 1)  # Example: batch_size=1, sequence_length=128, n_features=1
    
    # Convert PyTorch to ONNX
    convert_pytorch_to_onnx(pytorch_model_path, onnx_model_path, input_shape)
    
    # Convert ONNX to TensorFlow
    convert_onnx_to_tensorflow(onnx_model_path, tf_model_path)
    
    # Convert TensorFlow to TensorFlow Lite
    convert_tf_to_tflite(tf_model_path, tflite_model_path, input_shape)

if __name__ == "__main__":
    main() 