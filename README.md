 Instalación

Clonar el repositorio
git clone https://github.com/pablosantander1/Motosierra.git

Ingresar al directorio del proyecto
cd Motosierra

Crear el entorno virtual
python -m venv venv

Activar el entorno virtual Windows
venv\Scripts\activate

Instalar las dependencias

Actualizar pip
python -m pip install --upgrade pip
Instalar las librerías requeridas
pip install -r requirements.txt

(Opcional para una version mas acelerada de PyTorch con GPU NVIDIA compatile con CUDA 12.6
Instala PyTorch, TorchVision y TorchAudio con CUDA 12.6 
python -c "import torch; print(torch.cuda.is_available());print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU no disponible')"

Verificar Pytoch
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU no disponible')"
