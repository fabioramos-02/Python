import cv2
import os
import re
import easyocr
import json


from glob import glob
from tqdm import tqdm
from loguru import logger


class PlateDataAnalysis:
    def __init__(self) -> None:
        self.reader = easyocr.Reader(['en', 'pt'])  # Configura EasyOCR para inglês e português

    def convert_video_to_images(self, video_path: str, images_folder: str):
        cam = cv2.VideoCapture(video_path)

        try:
            # Verifica se o diretório de saída existe; caso contrário, cria
            if not os.path.exists(images_folder):
                os.makedirs(images_folder)
                first_time = True
            else:
                first_time = False
                logger.info(f'Path already exists: {images_folder}')

        except OSError:
            logger.error(f'Error: Creating directory of {images_folder}')

        if first_time:
            currentframe = 0
            while cam.isOpened():
                ret, frame = cam.read()

                if ret:
                    name = f'./{images_folder}/frame_{str(currentframe)}.jpg'
                    logger.info(f'Processing Frame: {currentframe}')

                    # Salva o frame como imagem
                    cv2.imwrite(name, frame)

                    # Avança 15 frames para capturar menos imagens
                    cam.set(cv2.CAP_PROP_POS_FRAMES, currentframe)
                    currentframe += 15
                else:
                    # Libera o recurso quando não há mais frames
                    cam.release()
                    break

        cam.release()
        cv2.destroyAllWindows()
        return True


    def is_valid_plate(self, plate: str) -> bool:
        """Valida se o texto corresponde ao formato de uma placa."""
        pattern = r'[A-Z]{3}[0-9][A-Z0-9][0-9]{2}'  # Regex para placas do Mercosul
        return re.match(pattern, plate) is not None

    def read_text_from_image(self, path_image: str) -> list:
        """Lê texto de uma imagem usando EasyOCR."""
        logger.info(f"Processando imagem: {path_image}")
        return self.reader.readtext(path_image)

    def filter_plates(self, text_items: list) -> list:
        """Filtra e retorna apenas os textos que são placas válidas."""
        plates = []
        for _, text, confidence in text_items:
            if self.is_valid_plate(text):
                plates.append({"plate": text, "confidence": confidence})
        return plates

    def list_images(self, path: str) -> list:
        """Lista todas as imagens em um diretório."""
        jpgs = glob(path)
        return jpgs

     
            
if __name__ == '__main__':
    # As opções de decodificação são 'greedy', 'beamsearch' e 'wordbeamsearch'
    decoder = 'beamsearch'

    # Instancia o objeto da classe PlateDataAnalysis
    plate_analysis = PlateDataAnalysis()

    # Converte o vídeo em imagens salvas na pasta 'images'
    plate_analysis.convert_video_to_images('./video_placa.mp4', 'images')

    # Lista todas as imagens extraídas
    images_list = plate_analysis.list_images('./images/*')

    # Inicializa a lista para armazenar as placas
    plates_list = {}

    # Processa cada imagem na lista
    for image in tqdm(images_list):
        # Extrai texto da imagem usando o método EasyOCR
        texts = plate_analysis.read_text_from_image(image, decoder)

        # Filtra os textos extraídos para identificar placas válidas
        text_plate = plate_analysis.filter_plates(texts)

        # Loga as placas encontradas
        logger.info(f"Texto da placa: {text_plate}")
