import os
import shutil
from typing import List, Tuple
from pathlib import Path
import re
from glob import glob
import csv
import argparse
from statistics import mean
import pandas as pd
import pypdf
from pdf2image import convert_from_path
from PIL import Image
from DECIMER import predict_SMILES_with_confidence
from rdkit import Chem
from rdkit.Chem import Draw
from decimer_segmentation import segment_chemical_structures_from_file
from fpdf import FPDF


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

        To set a confidence value for splitting:
        >>> args = create_parser()
        >>> print(args.c)

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
    parser.add_argument(
        "-c", help="Enter a value at which the confidences will be split"
    )
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


def get_conf_value(args: argparse.Namespace) -> str:
    """Extracts the confidence value from the argument.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        str: The confidence value given as an argument.

    Example:
        >>> args = create_parser()
        >>> conf_value = get_conf_value(args)
        >>> print(conf_value)
        '0.5'
    """
    confidence_value = args.c
    return confidence_value


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


def get_doi(pdf_name: str) -> str:
    """Reads a given PDF and extracts the DOI (Digital Object Identifier).

    Args:
        pdf_name (str): The filepath of the PDF.

    Returns:
        str: The extracted DOI.

    Example:
        >>> pdf_path = 'path/to/file.pdf'
        >>> doi = get_doi(pdf_path)
        >>> print(doi)
        '10.1234/example.doi'
    """
    pattern = r"\b10\.\d{0,9}\/[-._;()\/:a-zA-Z0-9]+\b"
    with open(pdf_name, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)
        content = ""
        for i in enumerate(pdf_reader.pages):
            page = pdf_reader.pages[i]
            content += page.extract_text()
    result = re.findall(pattern, content)[0]
    return result


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
    filename = Path(filepath).stem

    try:
        doi = get_doi(filepath)
    except IndexError:
        doi = filename
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
    pages = convert_from_path(filepath, 200)
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
            for segment_index in enumerate(segments):
                out_img = Image.fromarray(segments[segment_index])
                page_path = Path(image_path).stem
                out_img_path = os.path.join(
                    out_dir_path, f"{page_path}_{segment_index}_segmented.png"
                )
                out_img.save(out_img_path, "PNG")


def get_smiles_with_avg_confidence(filepath: str) -> None:
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
    for dir_name in [
        d for d in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, d))
    ]:
        newdirpath = os.path.join(dirpath, dir_name)
        for im in os.listdir(newdirpath):
            im_path = os.path.join(newdirpath, im)
            if im_path.endswith(".png"):
                smiles_with_confidence = predict_SMILES_with_confidence(im_path)
                smiles_characters = [item[0] for item in smiles_with_confidence]
                smiles = "".join(smiles_characters)
                confidence_list = [item[1] for item in smiles_with_confidence]
                avg_confidence = mean(confidence_list)
                data = [[smiles], [avg_confidence]]
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


def create_output_dictionary(filepath: str, doi: str) -> None:
    """Create an output dictionary containing SMILES and confidence scores for each segmented image.

    Args:
        filepath (str): Path to a PDF file with subdirectories containing segmented images.
        doi (str): DOI or filename of the PDF file.

    Returns:
        dict: Output dictionary with DOI/Filename as keys and image IDs, SMILES, and confidence scores as values.

    Example:
        >>> create_output_dictionary('path/to/example.pdf', 'example_doi')

        This will create an output dictionary containing SMILES and confidence scores for each segmented image.
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
                        confidence = csv_contents[1]
                        im_id = Path(newfilepath).stem
                        key = im_id[:-10]
                        im_dict[key] = smiles
                        im_dict[key].append(confidence)
                        output_dict[doi].update(im_dict)
    return output_dict


def custom_sort(image_id):
    parts = image_id.split("_")
    return int(parts[1]), int(parts[2])


def create_output_csv(filepath: str, output_dict: dict) -> None:
    filename = Path(filepath).stem
    dirpath = Path(filepath).parent
    first_level_keys = list(output_dict.keys())
    data_rows = []
    for first_level_key in first_level_keys:
        second_level_keys = list(output_dict[first_level_key].keys())
        for second_level_key in second_level_keys:
            smiles = output_dict[first_level_key][second_level_key][0]
            confidence_score = output_dict[first_level_key][second_level_key][1][0]
            data_rows.append(
                [first_level_key, second_level_key, smiles, confidence_score]
            )
    df_doi_imid_smiles_conf = pd.DataFrame(
        data_rows,
        columns=[
            "DOI/Filename",
            "Image ID",
            "Predicted Smiles",
            "Avg Confidence Score",
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


def get_predicted_images(output_df_with_errors):
    indexes_with_none = output_df_with_errors[
        output_df_with_errors.isna().any(axis=1)
    ].index
    for idx in indexes_with_none:
        im_id = output_df_with_errors.loc[idx, "Image ID"]
        output_df_with_errors.loc[idx, "SMILES Error"] = f"{im_id}_predicted.png"
        output_df_with_errors_and_pred = output_df_with_errors
    return output_df_with_errors_and_pred


def create_pdf(filepath, output_df_with_errors_and_pred_im):
    """creates a pdf containing image name SMILES confidence score as well as the segmented image and the predicted image or SMILES parse error"""
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
        predicted_image_path = os.path.join(subdirpath, predicted_image_name)
        if "SMILES Parse Error" not in predicted_image_path:
            pdf.image(predicted_image_path, x=120, y=60, w=100)
            if row["Avg Confidence Score"]:
                pdf.cell(0, 10, txt=row["Avg Confidence Score"], ln=True)
        else:
            pdf.cell(0, 10, txt=predicted_image_name, ln=True)
            if row["Avg Confidence Score"]:
                pdf.cell(0, 10, txt=row["Avg Confidence Score"], ln=True)
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


def split_good_bad(csv_path: str, value: float = 0.9) -> None:
    """Split a CSV based on confidence scores.

    This function reads a CSV file containing confidence scores and splits it into two CSV files:
    one for confidence scores greater than or equal to a specified value, and one for scores below the value.

    Args:
        csv_path (str): Path to the CSV file containing confidence scores.
        value (float, optional): Threshold value for confidence scores. Defaults to 0.9.

    Returns:
        None

    Example:
        >>> split_good_bad('path/to/confidence_scores.csv', value=0.8)

        This will split the CSV file into two based on confidence scores (good and bad) and create new CSV files.
    """
    merged_file = pd.read_csv(csv_path)
    path = Path(csv_path)
    parent_path = path.parent.absolute()
    good = merged_file.loc[merged_file["Avg Confidence Score"] >= value]
    bad = merged_file.loc[merged_file["Avg Confidence Score"] < value]
    output_good = os.path.join(parent_path, f"avg_conf_higher_than_{value}.csv")
    output_bad = os.path.join(parent_path, f"avg_conf_lower_than_{value}.csv")
    good.to_csv(output_good)
    bad.to_csv(output_bad)


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
    pdflist, directory = get_list_of_files(args)
    conf_value = get_conf_value(args)
    if conf_value is None:
        conf_value = 0.9
    abs_path_input = os.path.abspath(directory)
    output_name = [f for f in os.listdir(directory) if f.endswith(".txt")]
    terminal_output = os.path.join(abs_path_input, output_name[0])
    print(pdflist)
    for pdf in pdflist:
        print(pdf)
        doi = get_doi_from_file(pdf)
        print(doi)
        create_output_directory(pdf)
        get_single_pages(pdf)
        get_segments(pdf)
        get_smiles_with_avg_confidence(pdf)
        out_dict = create_output_dictionary(pdf, doi)
        df_doi_imid_smiles_conf, out_csv_path = create_output_csv(pdf, out_dict)
        output_df_with_errors = get_parse_error(
            df_doi_imid_smiles_conf, out_csv_path, terminal_output
        )
        output_df_with_errors_and_pred_im = get_predicted_images(
            pdf, output_df_with_errors
        )
        create_pdf(pdf, output_df_with_errors_and_pred_im)
        move_pdf(pdf)
    csv_path = concatenate_csv_files(abs_path_input, output_file="merged_output.csv")
    split_good_bad(csv_path, conf_value)


if __name__ == "__main__":
    main()
