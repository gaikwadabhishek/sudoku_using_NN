import numpy as np
import pandas as pd

from keras import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Input
from keras.utils import to_categorical
from keras.models import model_from_json

def load_data(nb_train=40000, nb_test=10000, full=False):

    if full:
        sudokus = pd.read_csv('sudoku.csv').values
    else:
        sudokus = next(
            pd.read_csv('sudoku.csv', chunksize=(nb_train + nb_test))
        ).values
        
    quizzes, solutions = sudokus.T
    flatX = np.array([np.reshape([int(d) for d in flatten_grid], (9, 9))
                      for flatten_grid in quizzes])
    flaty = np.array([np.reshape([int(d) for d in flatten_grid], (9, 9))
                      for flatten_grid in solutions])
    
    return (flatX[:nb_train], flaty[:nb_train]), (flatX[nb_train:], flaty[nb_train:])

def delete_digits(X, n_delet=1):
    """
    This function is used to create sudoku quizzes from solutions
    
    Parameters
    ----------
    X (np.array), shape (?, 9, 9, 9|10): input solutions grids.
    n_delet (integer): max number of digit to suppress from original solutions
    
    Returns
    -------
    grids: np.array of grids to guess in one-hot way. Shape: (?, 9, 9, 10)
    """
    grids = X.argmax(3)  # get the grid in a (9, 9) integer shape
    for grid in grids:
        grid.flat[np.random.randint(0, 81, n_delet)] = 0  # generate blanks (replace = True)
        
    return to_categorical(grids)


def batch_smart_solve(grids, solver):
    """
    NOTE : This function is ugly, feel free to optimize the code ...
    
    This function solves quizzes in the "smart" way. 
    It will fill blanks one after the other. Each time a digit is filled, 
    the new grid will be fed again to the solver to predict the next digit. 
    Again and again, until there is no more blank
    
    Parameters
    ----------
    grids (np.array), shape (?, 9, 9): Batch of quizzes to solve (smartly ;))
    solver (keras.model): The neural net solver
    
    Returns
    -------
    grids (np.array), shape (?, 9, 9): Smartly solved quizzes.
    """
    grids = grids.copy()
    for _ in range((grids == 0).sum((1, 2)).max()):
        preds = np.array(solver.predict(to_categorical(grids)))  # get predictions
        probs = preds.max(2).T  # get highest probability for each 81 digit to predict
        values = preds.argmax(2).T + 1  # get corresponding values
        zeros = (grids == 0).reshape((grids.shape[0], 81))  # get blank positions

        for grid, prob, value, zero in zip(grids, probs, values, zeros):
            if any(zero):  # don't try to fill already completed grid
                where = np.where(zero)[0]  # focus on blanks only
                confidence_position = where[prob[zero].argmax()]  # best score FOR A ZERO VALUE (confident blank)
                confidence_value = value[confidence_position]  # get corresponding value
                grid.flat[confidence_position] = confidence_value  # fill digit inplace
    return grids



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

from os import sys
from sudokuextract.extract import extract_sudoku, load_image, predictions_to_suduko_string

img = load_image(sys.argv[1])
predictions, sudoku_box_images, whole_sudoku_image = extract_sudoku(img)
numbers = predictions_to_suduko_string(predictions)

numbers = numbers.replace('\n','')
numbers = np.array(list((numbers)))
#numbers_float = np.array(numbers).astype(np.float32).reshape(-1,9,9)
numbers_int = np.array(numbers).astype(np.int32).reshape(9,9)
print(numbers_int)
numbers = numbers_int.reshape(-1,9,9)
#numbers = numbers.reshape(-1,9,9)
numbers = to_categorical(numbers).astype('float32')
smart_guesses = batch_smart_solve(numbers.argmax(3),loaded_model)
print(smart_guesses)
