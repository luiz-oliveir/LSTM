import sys
import os
import subprocess
from pathlib import Path

def setup_environment():
    project_dir = Path(__file__).parent
    venv_dir = project_dir / '.venv'
    data_dir = project_dir / 'data'
    
    # Create directories if they don't exist
    data_dir.mkdir(exist_ok=True)
    print(f"Ensuring data directory exists at: {data_dir}")
    
    # Create virtual environment if it doesn't exist
    if not venv_dir.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', str(venv_dir)], check=True)
    
    # Get the path to pip and python in the virtual environment
    if os.name == 'nt':  # Windows
        pip_path = venv_dir / 'Scripts' / 'pip.exe'
        python_path = venv_dir / 'Scripts' / 'python.exe'
    else:  # Unix/Linux/MacOS
        pip_path = venv_dir / 'bin' / 'pip'
        python_path = venv_dir / 'bin' / 'python'
    
    # Upgrade pip
    print("Upgrading pip...")
    subprocess.run([str(pip_path), 'install', '--upgrade', 'pip'], check=True)
    
    # Install requirements
    print("Installing requirements...")
    requirements_file = project_dir / 'requirements.txt'
    subprocess.run([str(pip_path), 'install', '-r', str(requirements_file)], check=True)
    
    # Run the main script using the virtual environment's Python
    print("\nRunning LSTM-VAE script...")
    main_script = project_dir / 'LSTM_VAE com ajustes.py'
    
    # Use subprocess.Popen to allow interactive input
    process = subprocess.Popen(
        [str(python_path), str(main_script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )
    
    # Monitor the output and handle input when needed
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            if "Enter your choice (1 or 2):" in output:
                # Automatically choose option 1 to generate sample data
                process.stdin.write("1\n")
                process.stdin.flush()
                print("1")  # Echo the choice
    
    # Check for errors
    rc = process.poll()
    if rc != 0:
        error = process.stderr.read()
        print("Error occurred:", error)
        sys.exit(rc)

if __name__ == '__main__':
    setup_environment()