# Docker Installation Guide for Windows

## Method 1: Docker Desktop (Recommended)

### Step 1: Download Docker Desktop
1. Go to https://www.docker.com/products/docker-desktop/
2. Click "Download for Windows"
3. Download Docker Desktop Installer.exe

### Step 2: System Requirements Check
**Before installing, ensure your system meets these requirements:**

**Windows 11/10 Requirements:**
- Windows 11 64-bit: Home or Pro version 21H2 or higher
- Windows 10 64-bit: Home or Pro 21H1 (build 19043) or higher
- Enable Hyper-V and Containers Windows features
- BIOS-level hardware virtualization support enabled

**Hardware Requirements:**
- 64-bit processor with Second Level Address Translation (SLAT)
- 4GB system RAM minimum (8GB+ recommended)
- Virtualization enabled in BIOS

### Step 3: Enable Required Windows Features

#### Option A: Via PowerShell (Run as Administrator)
```powershell
# Enable Hyper-V
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All

# Enable Windows Subsystem for Linux (WSL2)
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Restart required after these commands
Restart-Computer
```

#### Option B: Via GUI
1. Open "Turn Windows features on or off" (search in Start menu)
2. Check these boxes:
   - ✅ Hyper-V
   - ✅ Windows Subsystem for Linux
   - ✅ Virtual Machine Platform
   - ✅ Containers
3. Click OK and restart when prompted

### Step 4: Install WSL2 (Required for Docker Desktop)
```powershell
# Install WSL2
wsl --install

# Set WSL2 as default version
wsl --set-default-version 2

# Install Ubuntu (recommended)
wsl --install -d Ubuntu
```

### Step 5: Install Docker Desktop
1. Run the Docker Desktop Installer.exe
2. Follow the installation wizard
3. **Important**: Choose "Use WSL 2 instead of Hyper-V" when prompted
4. Complete installation and restart computer

### Step 6: Verify Installation
```powershell
# Check Docker version
docker --version

# Check Docker Compose version
docker-compose --version

# Test Docker installation
docker run hello-world
```

## Method 2: Docker via Chocolatey Package Manager

### Step 1: Install Chocolatey (if not installed)
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

### Step 2: Install Docker via Chocolatey
```powershell
# Install Docker Desktop
choco install docker-desktop

# Or install Docker Engine only
choco install docker-engine
```

## Method 3: Manual Docker Engine Installation (Advanced)

### For Windows Server or if you prefer Docker Engine without Desktop:
1. Download Docker Engine from: https://download.docker.com/win/static/stable/x86_64/
2. Extract the ZIP file
3. Add docker.exe to your PATH
4. Install Docker service manually

## Post-Installation Setup

### 1. Configure Docker Settings
Open Docker Desktop and configure:
- **Resources**: Allocate sufficient CPU/Memory
- **WSL Integration**: Enable for your WSL distributions
- **File Sharing**: Add drives you want to access from containers

### 2. Test Your Installation
```powershell
# Basic test
docker run hello-world

# Test with a simple web server
docker run -d -p 8080:80 nginx
# Visit http://localhost:8080 in browser

# Clean up test container
docker ps
docker stop <container_id>
docker rm <container_id>
```

### 3. Common Docker Commands
```powershell
# List running containers
docker ps

# List all containers
docker ps -a

# List images
docker images

# Stop all containers
docker stop $(docker ps -q)

# Remove all containers
docker rm $(docker ps -aq)

# Remove all images
docker rmi $(docker images -q)
```

## Troubleshooting Common Issues

### Issue 1: Virtualization Not Enabled
**Error**: "Docker Desktop requires Windows 10 Pro/Enterprise"
**Solution**: 
1. Enter BIOS/UEFI settings during boot
2. Enable "Intel VT-x" or "AMD-V" virtualization
3. Enable "Hyper-V" in Windows features

### Issue 2: WSL2 Installation Issues
**Error**: "WSL 2 installation is incomplete"
**Solution**:
```powershell
# Update WSL kernel
wsl --update

# Check WSL version
wsl -l -v

# If Ubuntu shows version 1, convert to version 2:
wsl --set-version Ubuntu 2
```

### Issue 3: Docker Desktop Won't Start
**Solutions**:
1. Restart Docker Desktop as Administrator
2. Reset to factory defaults in Docker Desktop settings
3. Reinstall Docker Desktop
4. Check Windows Event Viewer for detailed errors

### Issue 4: Permission Denied Errors
**Solution**:
```powershell
# Add your user to docker-users group
net localgroup docker-users "your-username" /add

# Logout and login again
```

## Next Steps: Run Your Fraud Detection Demo

Once Docker is installed, you can run the demo service:

```powershell
# Navigate to your project directory
cd "C:\Users\oumme\OneDrive\Desktop\FRAUD DETECTION\hhgtn-project"

# Build the Docker image
docker build -t fraud-detection-demo ./demo_service

# Run the container
docker run -p 8000:8000 fraud-detection-demo

# Or use docker-compose
docker-compose up --build
```

Visit http://localhost:8000 to see your demo interface!

## Additional Resources

- Docker Documentation: https://docs.docker.com/
- Docker Desktop for Windows: https://docs.docker.com/desktop/windows/
- WSL2 Documentation: https://docs.microsoft.com/en-us/windows/wsl/
- Docker Compose Documentation: https://docs.docker.com/compose/

## Quick Installation Script

Save this as `install-docker.ps1` and run as Administrator:

```powershell
# Quick Docker Installation Script for Windows
Write-Host "Installing Docker Desktop for Windows..."

# Enable required Windows features
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Download and install Docker Desktop (you'll need to download manually)
Write-Host "Please download Docker Desktop from: https://www.docker.com/products/docker-desktop/"
Write-Host "After installation, restart your computer and run: docker --version"
```
