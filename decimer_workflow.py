"""
DECIMER Workflow Script

This script implements a workflow for processing PDF publications to extract chemical structures,
predict SMILES (Simplified Molecular Input Line Entry System) representations with confidence scores,
and generate comprehensive output reports.

The script performs the following steps:
1. Parsing command-line arguments to specify input files or directories.
2. Image segmentation of PDF pages to isolate chemical structure images.
3. Prediction of SMILES representations with confidence scores for segmented images.
4. Creation of CSV files containing SMILES and confidence data for each segmented image.
5. Concatenation and analysis of CSV files to categorize chemical structures based on confidence scores.
6. Identification of parsing errors and generation of detailed output reports (PDFs) with segmented images
   and associated SMILES predictions or error messages.
7. Merging of output PDFs into comprehensive reports for each processed publication.

Command-line Usage:
- Use `-f` to specify a single PDF file for processing.
- Use `-folder` to specify a directory containing multiple PDF files for batch processing.
- Optionally, use `-c` to set a confidence threshold for splitting SMILES predictions.

Example:
$ python3 decimer_workflow.py -folder path/to/directory/with/publications -c desired_confidence_value_to_split &> path/to/directory/with/publications/out.txt 2>&1


Dependencies:
- Python 3.x
- DECIMER (https://decimer.ai/)
- PDF2Image (https://github.com/Belval/pdf2image)
- RDKit (https://github.com/rdkit/rdkit)
- FPDF (https://github.com/reingart/pyfpdf)

Author: Laurin Lederer
"""

import os
import shutil
from typing import List, Tuple
from pathlib import Path
import re
from glob import glob
import csv
import argparse
from statistics import mean
from datetime import datetime
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
from DECIMER import predict_SMILES
from rdkit import Chem
from rdkit.Chem import Draw
from decimer_segmentation import segment_chemical_structures_from_file
from fpdf import FPDF
from pdf2doi import pdf2doi

import MolNexTR


def create_parser():
    """Create an ArgumentParser for processing PDF and .smi files.

    This function sets up an ArgumentParser to handle command-line arguments for the script.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.

    Example:
        To process a single file:
        >>> args = create_parser()
        >>> print(args.f)

        To process all PDF files in a folder:
        >>> args = create_parser()
        >>> print(args.folder)

    Command-line Usage:
        Use either -f to run the script on a single file or -folder to run the script on all
        PDF files in a folder.

        Example:
        $ python script.py -f path/to/file.pdf
        $ python script.py -folder path/to/folder -c 0.5
    """
    parser = argparse.ArgumentParser(
        description="Use either -f to run the script on a single file oder -folder to run the script on all pdf files in a folder"
    )
    parser.add_argument("-f", help="Input relative or absolute path of a publication")
    parser.add_argument("-folder", help="Enter a path to a folder")

    parser.add_argument("-p", help="Which prediction model to use, defaults to DECIMER")
    parser.add_argument("-m", help="Specify the path to the MolNexTR get_model")
    parser.add_argument("-d", help="Specify the device to use (default: cpu)")
    args = parser.parse_args()
    return args


def get_filepath(args: argparse.Namespace) -> str:
    """Extracts the filepath from the argument.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        str: The filepath to the PDF file.

    Example:
        >>> args = create_parser()
        >>> filepath = get_filepath(args)
        >>> print(filepath)
        'path/to/file.pdf'
    """
    publication = args.f
    return publication


def get_list_of_files(args: argparse.Namespace) -> Tuple[List[str], str]:
    """Retrieve a list of .pdf files from the specified directory.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        tuple: A tuple containing:
            - list: A list of all PDF files in the given directory.
            - str: The path of the directory.

    Example:
        >>> args = create_parser()
        >>> pdf_files, directory = get_list_of_files(args)
        >>> print(pdf_files)
        ['path/to/file1.pdf', 'path/to/file2.pdf']
        >>> print(directory)
        'path/to/folder'
    """
    directory = args.folder
    if isinstance(directory, str):
        pdf_list = glob(os.path.join(directory, "*.{}".format("pdf")))
        return pdf_list, directory
    pdf_list = []
    return pdf_list, directory


def get_doi_from_file(filepath: str) -> str:
    """Extract DOI or filename from the given file path.

    Args:
        filepath (str): Path to a file.

    Returns:
        str: DOI if available, otherwise filename without extension.

    Example:
        >>> get_doi_from_file('path/to/example.pdf')

        This will return the DOI extracted from the PDF file if available,
        otherwise, it will return the filename without extension.
    """
    doi_dict = pdf2doi(filepath)
    doi = doi_dict["identifier"]
    if not doi:
        doi = Path(filepath).stem
    return doi


def create_output_directory(filepath: str) -> None:
    """Create the output directory based on the given file path.

    Args:
        filepath (str): Absolute or relative path to a file.

    Returns:
        None

    Example:
        >>> create_output_directory('path/to/example.pdf')

        This will create the directory 'path/to/example/' if it doesn't exist.
    """
    output_directory = filepath[:-4]
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    return output_directory


def get_single_pages(filepath: str) -> None:
    """Split the pages of a PDF file into individual .png images.

    Args:
        filepath (str): Absolute or relative path to a .pdf file.

    Returns:
        None

    Example:
        >>> get_single_pages('path/to/example.pdf')

        This will create individual .png images for each page in the 'path/to/example/' directory.
    """
    pages = convert_from_path(filepath, 96)
    for count, page in enumerate(pages):
        output_path = os.path.join(filepath[:-4], f"page_{count}.png")
        page.save(output_path, "PNG")


def get_segments(filepath: str) -> None:
    """Run DECIMER segmentation on each page of a PDF and save segmented images.

    This function goes through the created output directory, runs DECIMER segmentation on each page,
    creates a 'segments' directory for each page, and puts the segmented images inside those directories.

    Args:
        filepath (str): Absolute or relative path to a .pdf file.

    Returns:
        None

    Example:
        >>> get_segments('path/to/example.pdf')

        This will create segmented images for each page in the 'path/to/example_segments/' directories.
    """
    directory = filepath[:-4]
    filelist = os.listdir(directory)
    for image in filelist:
        if image.endswith(".png"):
            image_path = os.path.join(directory, image)
            out_dir_path = os.path.join(f"{image_path[:-4]}_segments")
            if not os.path.isdir(out_dir_path):
                os.mkdir(out_dir_path)
            segments = segment_chemical_structures_from_file(
                image_path, expand=True, poppler_path=None
            )
            for segment_index, img_array in enumerate(segments):
                out_img = Image.fromarray(img_array)
                page_path = Path(image_path).stem
                out_img_path = os.path.join(
                    out_dir_path, f"{page_path}_{segment_index}_segmented.png"
                )
                out_img.save(out_img_path, "PNG")


def copy_segments(filepath: str) -> None:
    directory = filepath[:-4]

    segments_dir = os.path.join(directory, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    for root, _, files in os.walk(directory):
        if root == segments_dir:
            continue

        for file in files:
            if file.endswith("segmented.png"):
                imgpath = os.path.join(root, file)
                imgname = os.path.basename(imgpath)
                copied_path = os.path.join(segments_dir, imgname)
                shutil.copyfile(imgpath, copied_path)


def get_molnextr_inp(args: argparse.Namespace):
    """Retrieve the path to the MolNexTR model and which device to use.
    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    Returns:
        str: The path to the MolNexTR model.
    Example:
        >>> args = create_parser()
        >>> model_path = get_model_path(args)
        >>> print(model_path)
        'path/to/model.pth'
    """
    model_path = args.m
    device = args.d
    if device is None:
        device = "cpu"
    if model_path:
        return model_path, device
    else:
        return "Please specify the path to the MolNexTR model."


def get_smiles_with_molnextr(filepath: str) -> dict:
    smiles_dict = {}
    segments_dir = os.path.join(os.path.splitext(filepath), "segments")
    print(segments_dir)

    for img in os.listdir(segments_dir):
        img_name = "_".join(os.path.basename(img).split("_")[:3])
        print(img)
        img_path = os.path.join(f"{segments_dir}", img)
        smiles = MolNexTR.get_predictions(
            img_path, atoms_bonds=False, smiles=True, predicted_molfile=False
        )

        smiles_dict[img_name] = smiles
    return smiles_dict


def get_smiles_with_DECIMER(filepath: str) -> List:
    """Predict SMILES and average confidence for segmented images in subdirectories.

    This function takes the path to a PDF file, loops through the affiliated directory with each subdirectory,
    predicts SMILES and an average confidence score for each segmented image.
    It also prints out a predicted image from the SMILES if possible.
    For each segmented image, the SMILES and confidence values will be written to an individual .csv file.

    Args:
        filepath (str): Path to a PDF file with subdirectories containing segmented images.

    Returns:
        None

    Example:
        >>> get_smiles_with_avg_confidence('path/to/example.pdf')

        This will predict SMILES and average confidence for segmented images in subdirectories
        and create .csv files and predicted images for each segmented image.
    """
    dirpath = filepath[:-4]
    smiles_list = []
    for dir_name in [
        d for d in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, d))
    ]:
        newdirpath = os.path.join(dirpath, dir_name)
        for im in os.listdir(newdirpath):
            im_path = os.path.join(newdirpath, im)
            if im_path.endswith(".png"):
                _, smiles = predict_SMILES(im_path)
                smiles_characters = [item[0] for item in smiles]
                smiles = "".join(smiles_characters)
                smiles_list.append(smiles)
                confidence_list = [item[1] for item in smiles]
                data = [[smiles]]
                csv_path = f"{im_path[:-13]}predicted.csv"
                with open(csv_path, "w", encoding="UTF-8") as csvfile:
                    csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerows(data)
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
                predicted_image_title = f"{im_path[:-13]}predicted.png"
                try:
                    Draw.MolToFile(mol, predicted_image_title)
                except ValueError:
                    pass
    return smiles_list


def create_output_dictionary(filepath: str, doi: str) -> dict:
    """Create an output dictionary containing SMILES and confidence scores for each segmented image.

    Args:
        filepath (str): Path to a PDF file with subdirectories containing segmented images.
        doi (str): DOI or filename of the PDF file.

    Returns:
        dict: Output dictionary with DOI/Filename as keys and image IDs, SMILES, and confidence scores as values.

    Example:
        >>> create_output_dictionary('path/to/example.pdf', 'example_doi')

    This function traverses through the directory structure created from segmented images stored in CSV files within subdirectories of the provided PDF filepath. It extracts SMILES and confidence scores from these CSV files and organizes them into a dictionary.

    The output dictionary has the following structure:
    {
        'doi': {
            'image_id': ['smiles', 'confidence_score']
        }
    }

    Note:
    - The `filepath` should point to a PDF file with the same name as the provided `doi`, containing subdirectories with CSV files containing SMILES and confidence scores.
    - The `doi` parameter should be unique and can be used as a key in the output dictionary.
    - Each image is uniquely identified by its `image_id`, derived from the filename of the CSV file.
    - SMILES and confidence scores are extracted from the first and second rows of each CSV file, respectively.
    - If multiple images share the same `image_id`, only the last SMILES and confidence score encountered are stored in the output dictionary.

    """
    output_dict = {doi: {}}
    dirpath = filepath[:-4]
    for root, dirs, files in os.walk(dirpath):
        dirs.sort()
        for subdir in dirs:
            newdirpath = os.path.join(root, subdir)
            newdir_list = os.listdir(newdirpath)
            if newdir_list:
                newdir_list.sort()
            for subfiles in newdir_list:
                newfilepath = os.path.join(newdirpath, subfiles)
                if newfilepath.endswith(".csv"):
                    with open(newfilepath, "r", encoding="UTF-8") as csvfile:
                        csv_reader = csv.reader(csvfile)
                        im_dict = {}
                        csv_contents = []
                        for row in csv_reader:
                            csv_contents.append(row)
                        smiles = csv_contents[0]
                        im_id = Path(newfilepath).stem
                        key = im_id[:-10]
                        im_dict[key] = smiles
                        output_dict[doi].update(im_dict)
    return output_dict


def custom_sort(image_id: str) -> tuple:
    """Custom sorting function for sorting image IDs based on numerical parts.

    Args:
        image_id (str): Image ID in the format 'prefix_num1_num2'.

    Returns:
        tuple: Tuple containing two integers representing the numerical parts of the image ID.

    Example:
        >>> custom_sort('prefix_10_2')
        (10, 2)

    This function is used as a key function for sorting image IDs based on their numerical parts. It extracts the numerical parts from the provided image ID string and returns them as a tuple. The sorting is primarily used to ensure a consistent order of image IDs when creating the output CSV file.
    """
    parts = image_id.split("_")
    return int(parts[1]), int(parts[2])


def create_output_csv(filepath: str, output_dict: dict) -> tuple:
    """Create an output CSV file containing DOI/Filename, image IDs, SMILES, and confidence scores.

    Args:
        filepath (str): Path to the PDF file.
        output_dict (dict): Output dictionary containing SMILES and confidence scores for each segmented image.

    Returns:
        tuple: Tuple containing a pandas DataFrame with the data and the path to the output CSV file.

    Example:
        >>> create_output_csv('path/to/example.pdf', {'example_doi': {'image_id': ['smiles', 'confidence_score']}})

    This function generates an output CSV file containing the extracted SMILES and confidence scores for each segmented image. It utilizes the provided output dictionary to organize the data and writes it to a CSV file.

    The CSV file contains the following columns:
    - DOI/Filename: The DOI or filename of the PDF file.
    - Image ID: Unique identifier for each segmented image.
    - Predicted Smiles: SMILES representation of the predicted chemical structure.
    - Avg Confidence Score: Average confidence score associated with the predicted SMILES.

    Note:
    - The output CSV file is named based on the filename of the provided PDF file and saved in the same directory.
    """
    filename = Path(filepath).stem
    dirpath = Path(filepath).parent
    first_level_keys = list(output_dict.keys())
    data_rows = []
    for first_level_key in first_level_keys:
        second_level_keys = list(output_dict[first_level_key].keys())
        for second_level_key in second_level_keys:
            smiles = output_dict[first_level_key][second_level_key][0]
            data_rows.append([first_level_key, second_level_key, smiles])
    df_doi_imid_smiles_conf = pd.DataFrame(
        data_rows,
        columns=[
            "DOI/Filename",
            "Image ID",
            "Predicted Smiles",
        ],
    )
    df_doi_imid_smiles_conf["sort_key"] = df_doi_imid_smiles_conf["Image ID"].apply(
        custom_sort
    )
    df_doi_imid_smiles_conf = df_doi_imid_smiles_conf.sort_values(by="sort_key").drop(
        columns="sort_key"
    )
    out_csv_path = os.path.join(dirpath, filename, f"{filename}_out.csv")
    df_doi_imid_smiles_conf.to_csv(out_csv_path, index=False)
    return df_doi_imid_smiles_conf, out_csv_path


def get_parse_error(output_df, out_csv_path: str, terminal_output: str) -> None:
    """Get parsing errors from RDKit for each SMILES in a merged CSV file (if an error occurs).

    This function reads a merged CSV file containing SMILES predictions and checks for parsing errors
    based on the terminal output. It updates the CSV with a new column 'SMILES Error' containing the error messages.

    Args:
        merged_csv (str): Path to the merged CSV file.
        terminal_output (str): Path to the terminal output file.

    Returns:
        None

    Example:
        >>> get_parse_error('path/to/merged_output.csv', 'path/to/rdkit_terminal_output.txt')

        This will update 'merged_output.csv' with a new column 'SMILES Error' based on parsing errors.
    """
    with open(terminal_output, "r") as file:
        ter_output = file.readlines()
    all_errors = []
    for smiles in output_df["Predicted Smiles"]:
        matching = [s for s in ter_output if smiles in s]
        if matching:
            msg = matching[0]
            word1 = "SMILES"
            word2_options = ("parsing", "input")
            pattern = re.compile(
                f'{re.escape(word1)}(.*?)(?:{ "|".join(map(re.escape, word2_options)) })',
                re.DOTALL,
            )
            match = pattern.search(msg)
            if match:
                error_msg = match.group(0)
        else:
            error_msg = None
        all_errors.append(error_msg)
    output_df["SMILES Error"] = all_errors
    output_df_with_errors = output_df
    output_df_with_errors.to_csv(out_csv_path)
    return output_df_with_errors


def get_predicted_images(output_df_with_errors: pd.DataFrame) -> pd.DataFrame:
    """Assign predicted image names for rows with missing values in the output DataFrame.

    Args:
        output_df_with_errors (pd.DataFrame): DataFrame containing rows with missing values.

    Returns:
        pd.DataFrame: DataFrame with predicted image names assigned to rows with missing values.

    This function identifies rows in the input DataFrame where at least one column contains a missing value (NaN). It then generates predicted image names based on the 'Image ID' column and assigns them to the corresponding rows. The predicted image name is constructed by appending '_predicted.png' to the 'Image ID'.

    Note:
    - The input DataFrame should contain the 'Image ID' column.
    - Rows with missing values are identified using the 'isna()' method.
    - Predicted image names are assigned to the 'SMILES Error' column.
    """
    indexes_with_none = output_df_with_errors[
        output_df_with_errors.isna().any(axis=1)
    ].index
    for idx in indexes_with_none:
        im_id = output_df_with_errors.loc[idx, "Image ID"]
        output_df_with_errors.loc[idx, "SMILES Error"] = f"{im_id}_predicted.png"
    output_df_with_errors_and_pred = output_df_with_errors
    return output_df_with_errors_and_pred


def create_pdf(filepath: str, output_df_with_errors_and_pred_im: pd.DataFrame) -> None:
    """Create a PDF containing image names, SMILES, confidence scores, segmented images, and predicted images/SMILES parse errors.

    Args:
        filepath (str): Path to the PDF file.
        output_df_with_errors_and_pred_im (pd.DataFrame): DataFrame containing data for generating the PDF.

    Returns:
        None

    This function generates a PDF containing information about segmented images, predicted SMILES, confidence scores, and corresponding images or SMILES parse errors. Each page of the PDF displays one segmented image along with its predicted SMILES or error message.

    Note:
    - The PDF is created using the FPDF library.
    - Each page of the PDF corresponds to a row in the input DataFrame.
    - The DataFrame should contain columns 'Image ID', 'Predicted Smiles', 'SMILES Error', and 'Avg Confidence Score'.
    """
    dirpath = os.path.splitext(filepath)[0]
    pdf = FPDF(orientation="L")
    pdf.set_font("Arial", size=12)
    for idx, row in output_df_with_errors_and_pred_im.iterrows():
        pagename = row["Image ID"]
        pagesplit = pagename.rsplit("_", 1)
        subdirpath = os.path.join(dirpath, f"{pagesplit[0]}_segments")
        pdf.add_page()
        segmented_image_path = os.path.join(subdirpath, f"{pagename}_segmented.png")
        pdf.cell(0, 10, txt=pagename)
        pdf.image(segmented_image_path, x=10, y=60, w=70)
        pdf.set_x(40)
        pdf.cell(0, 10, txt=row["Predicted Smiles"], ln=True, align="L")
        predicted_image_name = row["SMILES Error"]
        if "SMILES Parse Error" not in predicted_image_name:
            predicted_image_path = os.path.join(subdirpath, predicted_image_name)
            if os.path.exists(predicted_image_path):
                pdf.image(predicted_image_path, x=120, y=60, w=100)
        else:
            pdf.cell(0, 10, txt=predicted_image_name, ln=True)
    file_name = os.path.join(dirpath, "output.pdf")
    pdf.output(file_name)


def concatenate_csv_files(
    parent_directory: str, output_file: str = "merged_output.csv"
) -> str:
    """Concatenate CSV files for all publications from subdirectories.

    This function creates a CSV file that is concatenated for all publications from the previously
    concatenated CSV for each publication.

    Args:
        parent_directory (str): Input directory path where the PDFs are. It will also be the output directory
                               where the merged CSV will be saved.
        output_file (str, optional): Name to write the output CSV. Defaults to 'merged_output.csv'.

    Returns:
        str: Filepath of the merged CSV file.

    Example:
        >>> concatenate_csv_files('path/to/pdfs')

        This will concatenate CSV files for all publications from subdirectories and create 'merged_output.csv'.
    """
    subdirectories = [
        d
        for d in os.listdir(parent_directory)
        if os.path.isdir(os.path.join(parent_directory, d))
    ]
    dataframes = []
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(parent_directory, subdirectory)
        csv_files = [f for f in os.listdir(subdirectory_path) if f.endswith(".csv")]
        if csv_files:
            for csv_file in csv_files:
                csv_file_path = os.path.join(subdirectory_path, csv_file)
                df = pd.read_csv(csv_file_path)
                dataframes.append(df)
    merged_data = pd.concat(dataframes)
    merged_data.reset_index(drop=True, inplace=True)
    output_file_path = os.path.join(parent_directory, output_file)
    merged_wo_col0 = merged_data.iloc[:, 1:]
    merged_wo_col0.to_csv(output_file_path, index=True)
    return output_file_path


def concatenate_csv_files(
    parent_directory: str, output_file: str = "merged_output.csv"
) -> str:
    """Concatenate CSV files for all publications from subdirectories.

    This function creates a CSV file that is concatenated for all publications from the previously
    concatenated CSV for each publication.

    Args:
        parent_directory (str): Input directory path where the PDFs are. It will also be the output directory
                               where the merged CSV will be saved.
        output_file (str, optional): Name to write the output CSV. Defaults to 'merged_output.csv'.

    Returns:
        str: Filepath of the merged CSV file.

    Example:
        >>> concatenate_csv_files('path/to/pdfs')

        This will concatenate CSV files for all publications from subdirectories and create 'merged_output.csv'.
    """
    subdirectories = [
        d
        for d in os.listdir(parent_directory)
        if os.path.isdir(os.path.join(parent_directory, d))
    ]
    dataframes = []
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(parent_directory, subdirectory)
        csv_files = [f for f in os.listdir(subdirectory_path) if f.endswith(".csv")]
        if csv_files:
            for csv_file in csv_files:
                csv_file_path = os.path.join(subdirectory_path, csv_file)
                df = pd.read_csv(csv_file_path)
                dataframes.append(df)
    merged_data = pd.concat(dataframes)
    merged_data.reset_index(drop=True, inplace=True)
    output_file_path = os.path.join(parent_directory, output_file)
    merged_wo_col0 = merged_data.iloc[:, 1:]
    merged_wo_col0.to_csv(output_file_path, index=True)
    return output_file_path


def move_pdf(filepath: str) -> None:
    """Move a PDF file to its own directory.

    Args:
        filepath (str): Path to the PDF file.

    Returns:
        None

    Example:
        >>> move_pdf('path/to/example.pdf')

        This will move the 'example.pdf' file to a directory named 'example'.
    """
    output_dir = filepath[:-4]
    file_stem = Path(filepath).stem
    pdf_out = f"{file_stem}.pdf"
    output_path = os.path.join(output_dir, pdf_out)
    shutil.move(filepath, output_path)


def get_time_per_publication(input_path: str, time_list: list, pdf_list: list) -> None:
    """Calculate time differences between PDF files based on provided timestamps.

    This function calculates the time difference between consecutive timestamps
    in the given `time_list` and associates each time difference with its corresponding
    PDF file in `pdf_list`. The calculated time differences are then saved to a CSV
    file named 'times.csv' in the specified `input_path`.

    Args:
        input_path (str): Path to the input directory containing PDF files.
        time_list (list): List of timestamp strings in the format '%H:%M:%S'.
        pdf_list (list): List of PDF file names corresponding to the timestamps.

    Returns:
        None

    Example:
        >>> get_times('/path/to/input', ['14:40:09', '14:41:23', '14:41:43'], ['example1.pdf', 'example2.pdf'])

        This will calculate the time differences between timestamps and save the results
        to a CSV file 'times.csv' in the '/path/to/input' directory.

    """
    csv_out_path = os.path.join(input_path, "times.csv")
    time_dict = {}

    for idx, pdf in enumerate(pdf_list):
        start_time = datetime.strptime(time_list[idx], "%H:%M:%S")
        if idx + 1 < len(time_list):
            end_time = datetime.strptime(time_list[idx + 1], "%H:%M:%S")
            delta = end_time - start_time
            time_dict[pdf] = delta

    with open(csv_out_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["PDF File", "Time Difference"])

        for key, value in time_dict.items():
            writer.writerow([key, str(value)])


def extract_absolute_times() -> str:
    """Extract the current absolute time in the '%H:%M:%S' format.

    This function retrieves the current time using the system clock and
    formats it as a string in the '%H:%M:%S' format, representing hours,
    minutes, and seconds.

    Returns:
        str: A string containing the current time in the '%H:%M:%S' format.

    Example:
        >>> extract_absolute_times()
        '15:30:45'

    """
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time


def main():
    """Execute the entire program, performing segmentation, prediction, CSV creation, and PDF merging for chemical structure images.

    Steps:
    1. Parse command-line arguments.
    2. Identify input PDF files or directories.
    3. Set confidence threshold for SMILES prediction.
    4. Perform image segmentation, SMILES prediction, and CSV creation for each PDF.
    5. Concatenate CSV files and split structures based on confidence.
    6. Identify parsing errors and create an output PDF for each publication with segmented and predicted images or error messages.
    7. Merge PDFs for each finished publication directory.

    Note:
    This function assumes the availability of specific functions for each step in the process.

    Example:
    >>> main()
    """
    args = create_parser()
    pdf_list, directory = get_list_of_files(args)
    abs_path_input = os.path.abspath(directory)
    output_name = [f for f in os.listdir(directory) if f.endswith(".txt")]
    terminal_output = os.path.join(abs_path_input, output_name[0])
    time_list = []
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    time_list.append(current_time)
    for pdf in pdf_list:
        doi = get_doi_from_file(pdf)
        create_output_directory(pdf)
        get_single_pages(pdf)
        get_segments(pdf)
        if args.p == "molnextr":
            model, device = get_molnextr_inp(args)
            smiles_list = get_smiles_with_molnextr(pdf, model, device)
        else:
            smiles_list = get_smiles_with_DECIMER(pdf)
        out_dict = create_output_dictionary(pdf, doi)
        df_doi_imid_smiles_conf, out_csv_path = create_output_csv(pdf, out_dict)
        output_df_with_errors = get_parse_error(
            df_doi_imid_smiles_conf, out_csv_path, terminal_output
        )
        output_df_with_errors_and_pred_im = get_predicted_images(output_df_with_errors)
        create_pdf(pdf, output_df_with_errors_and_pred_im)
        move_pdf(pdf)
        current_time = extract_absolute_times()
        time_list.append(current_time)
    get_time_per_publication(abs_path_input, time_list, pdf_list)


if __name__ == "__main__":
    main()
