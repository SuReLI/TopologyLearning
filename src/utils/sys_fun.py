import os
import shutil
import datetime
import warnings
import cv2
import numpy as np
from PIL import Image


def create_dir(dir_name):
    if os.path.isdir(dir_name):
        return
    dir_parts = dir_name.split("/")
    directory_to_create = ""
    for part in dir_parts:
        directory_to_create += part + "/"
        if not os.path.isdir(directory_to_create):
            try:
                os.mkdir(directory_to_create)
            except:
                print("failed to create dir " + str(directory_to_create))
                raise Exception


def empty_dir(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_output_directory():
    """
    Create a global output directory that respect the following:
     - The output directory is in <Project name>/output/
     - The output directory is brand new so new outputs will not overwrite old ones.
     - The name of the output is like <OutputID>_DD-MM-YY_HH:MM
    """
    now = datetime.datetime.now()
    base_title = "_" + str(now.day) + "-" + str(now.month) + "-" + str(now.year) + "_" + str(now.hour) + ":" + \
                 str(now.minute) + ":" + str(now.second)
    cwd = os.getcwd()

    output_directory = cwd + "/outputs/"
    output_id = 0
    for filename in os.listdir(output_directory):
        current_output_id = -1
        try:
            words_list = filename.split("_")
            if len(words_list) > 1:
                current_output_id = int(words_list[1])
        except ValueError:
            # warnings.warn("file with name \"" + str(filename) + "\" in output directory have an unknown format.")
            continue  # The current dir is not an output directory so we ignore it
        if current_output_id >= output_id:
            output_id = current_output_id + 1
    output_directory += ('out_' + str(output_id) + base_title + "/")
    return output_directory


def generate_video(images, output_directory: str, filename):
    if len(filename) < 4 or filename[-4:] != ".mp4":
        filename += ".mp4"

    create_dir(output_directory)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, channels = images[0].shape
    fps = 30

    out = cv2.VideoWriter(output_directory + filename, fourcc, fps, (width, height))
    for image_data in images:
        image_data = image_data.astype(np.uint8)
        out.write(image_data)
    out.release()


def save_image(image: np.ndarray, directory, file_name):
    image = Image.fromarray(image)
    create_dir(directory)
    if not file_name.endswith(".png"):
        if len(file_name.split(".")) > 1:
            file_name = "".join(file_name.split(".")[:-1])  # Remove the last extension
        file_name += ".png"
    image.save(directory + file_name)


def get_red_green_color(value, hexadecimal=True):
    """
    Retourne une couleur correspondant à un gradient entre rouge (0) et vert (1) pour une valeur donnée entre 0 et 1
    :param value: valeur entre 0 et 1 définissant la couleur à récupérer
    """
    low_color = [255, 0, 0]
    high_color = [0, 255, 0]
    if hexadecimal:
        result = "#"
    else:
        result = []
    for index, (low, high) in enumerate(zip(low_color, high_color)):
        difference = high - low
        if hexadecimal:
            final_color = hex(int(low + value * difference))[2:]
            result += "0" + final_color if len(final_color) == 1 else final_color
        else:
            final_color = int(low + value * difference)
            result.append(final_color)
    return result
