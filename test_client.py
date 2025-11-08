import requests
import sys

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    response = requests.get('http://localhost:5000/health')
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}\n")


def test_classes():
    """Test the classes endpoint"""
    print("Testing classes endpoint...")
    response = requests.get('http://localhost:5000/classes')
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}\n")


def test_predict(audio_file_path, model='all'):
    """Test the predict endpoint"""
    print(f"Testing predict endpoint with file: {audio_file_path}")
    print(f"Using model: {model}")
    
    try:
        with open(audio_file_path, 'rb') as f:
            files = {'file': f}
            data = {'model': model}
            response = requests.post('http://localhost:5000/predict', files=files, data=data)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}\n")
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== Prediction Results ===")
            for model_name, prediction in result['predictions'].items():
                print(f"{model_name.upper()}: {prediction['predicted_class']} (Confidence: {prediction['confidence']}%)")
    
    except FileNotFoundError:
        print(f"Error: File not found - {audio_file_path}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == '__main__':
    # Test health endpoint
    test_health()
    
    # Test classes endpoint
    test_classes()
    
    # Test predict endpoint
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        model_type = sys.argv[2] if len(sys.argv) > 2 else 'all'
        test_predict(audio_file, model_type)
    else:
        print("Usage for prediction test:")
        print("  python test_client.py <audio_file_path> [model_type]")
        print("\nExample:")
        print("  python test_client.py path/to/audio.wav all")
        print("  python test_client.py path/to/audio.wav ann")
        print("\nModel types: ann, cnn1d, cnn2d, all (default)")
