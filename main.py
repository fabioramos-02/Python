import cv2
import os
import re
import easyocr
import json


from glob import glob
from tqdm import tqdm
from loguru import logger

import shutil

class PlateDataAnalysis:
    def __init__(self) -> None:
        self.reader = easyocr.Reader(['en', 'pt'])  # Configura EasyOCR para inglês e português

    def convert_video_to_images(self, video_path: str, images_folder: str, frame_skip: int = 10) -> bool:
        cam = cv2.VideoCapture(video_path)

        try:
            # Se a pasta já existir, exclui e recria
            if os.path.exists(images_folder):
                logger.info(f"Deleting existing folder: {images_folder}")
                shutil.rmtree(images_folder)  # Remove a pasta e todo o conteúdo
            os.makedirs(images_folder)  # Cria a nova pasta

        except OSError as e:
            logger.error(f"Erro ao manipular o diretório {images_folder}: {e}")
            return False

        currentframe = 0
        while cam.isOpened():
            ret, frame = cam.read()

            if ret:
                # Salva o frame como imagem na nova pasta
                name = f'./{images_folder}/frame_{str(currentframe)}.jpg'
                logger.info(f'Processando Frame: {currentframe}')
                cv2.imwrite(name, frame)
                cam.set(cv2.CAP_PROP_POS_FRAMES, currentframe)
                currentframe += frame_skip
            else:
                cam.release()
                break

        cam.release()
        return True


    def is_valid_plate(self, plate: str) -> bool:
        """Valida se o texto corresponde ao formato de uma placa."""
        pattern = r"^[A-Z]{3}[0-9][0-9A-Z][0-9]{2}$"
        return bool(re.fullmatch(pattern, plate))

    def read_text_from_image(self, path_image: str, decoder: str) -> list:
        """Lê texto de uma imagem usando EasyOCR."""
        try:
            result = self.reader.readtext(path_image, decoder=decoder)
            return result
        except Exception as e:
            logger.error(e)
            return None

    def filter_plates(self, text_items: list) -> list:
        """Filtra e retorna apenas os textos que são placas válidas."""
        for item in text_items:
            text = item[1].replace('-', '').replace('', '').upper()
            precision = item[2]
            
            logger.info(f"Texto extraido: {text} precisão: {precision}")
            
            is_plate = self.is_valid_plate(text)
            if precision > 0.75 and is_plate:
                data = {
                    "plate": text,
                    "precision": precision
                }
                return data
        return None

    def list_images(self, path: str) -> list:
        """Lista todas as imagens em um diretório."""
        jpgs = glob(path)
        return jpgs

     
            
if __name__ == '__main__':
    # As opções de decodificação são 'greedy', 'beamsearch' e 'wordbeamsearch'
    decoder = 'beamsearch'

    # Instancia o objeto da classe PlateDataAnalysis
    plate_analysis = PlateDataAnalysis()
    
    video_path = './teste.mp4'
    images_folder = 'images'
    frames_skip = 50

    # Converte o vídeo em imagens salvas na pasta 'images'
    plate_analysis.convert_video_to_images(video_path, images_folder, frames_skip)

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

        # Adiciona a placa à lista de placas, se válida
        if text_plate:
            plate = text_plate['plate']
            precision = text_plate['precision']

            if precision > plates_list.get(plate, 0):
                plates_list[plate] = precision
                logger.info(f"Placa adicionada: {plate} (Precisão: {precision:.2f})")
            else:
                logger.info(f"Placa {plate} já registrada com precisão maior ou igual.")
        else:
            logger.info(f"Nenhuma placa válida encontrada na imagem: {image}")
        
    with open(f"./plates_{decoder}.json", 'w') as file:
        json.dump(plates_list, file, indent=4)
    
    logger.info(f"Placas encontradas: {plates_list}")
