import os


def makedir(path: str, folders: list or str) -> str:
    if isinstance(folders, str):
        folders = [folders]
    temp_path, final_path = path, None
    for folder in folders:
        final_path = os.path.join(temp_path, folder)
        if folder not in os.listdir(temp_path):
            os.mkdir(final_path)
        temp_path = final_path
    return final_path


def listdir(path: str, check_in_folders: bool = True) -> list:
    """Set check_in_folders=True if you want to include also all files in the folders, False otherwise."""
    list_file = []
    for file_or_folder in os.listdir(path):
        file = os.path.join(path, file_or_folder)
        if os.path.isdir(file):                     # if it is a folder
            if check_in_folders:                    # include all files in all folders
                list_file.append(listdir(file))
        elif os.path.isfile(file):                  # if it is a file (or list of files)
            list_file.append(file)
    return list_file                                # list of all files (not file names but full path)


def flatten_list(input_list: list):
    final_list = []
    for temp_list in input_list:
        if isinstance(temp_list, list):
            temp_list = flatten_list(temp_list)
            for element in temp_list:
                final_list.append(element)
        else:
            element = temp_list
            final_list.append(element)
    return final_list


def listdir_flatten(path: str):
    file_list = listdir(path)
    return flatten_list(file_list)


def directory(file: str) -> str:
    return os.path.split(file)[0]


def file_name(file: str, keep_extension: bool = True) -> str:
    fname = os.path.split(file)[-1]  # Equal to: fname = os.path.basename(file)
    if keep_extension:
        return fname
    else:
        return os.path.splitext(fname)[0]


def file_extension(file: str) -> str:
    return os.path.splitext(file)[-1]


def check_extension(file: str, extension: str) -> bool:
    return file_extension(file) == extension


def check_keyword(file: str, keyword: str) -> bool:
    if file_name(file).find(keyword) != -1:
        return True
    else:
        return False


def keep_list_extension(list_files: list, extension: str, empty_error: bool = True) -> list or None:
    new_list = list(filter(lambda f: check_extension(f, extension), list_files))
    if not new_list:
        if empty_error:
            raise ValueError('There is no file, in the given list, with "' + extension + '" extension.')
        return None
    return new_list


def keep_list_keyword(list_files: list, keyword: str, empty_error: bool = True) -> list or None:
    new_list = list(filter(lambda f: check_keyword(f, keyword), list_files))
    if not new_list:
        if empty_error:
            raise ValueError('There is no file, in the given list, with keyword "' + keyword + '".')
        return None
    return new_list


def move_file(old_file: str, old_dir: str, new_dir: str) -> str or None:
    if not isinstance(old_file, str) or not (os.path.isfile(old_file) and old_file[:len(old_dir)] == old_dir):
        return None
    new_file = os.path.join(new_dir, old_file[len(old_dir):])
    os.makedirs(os.path.dirname(new_file), exist_ok=True)
    os.replace(old_file, new_file)
    return new_file


def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)
