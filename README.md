# Face recognition
* Pipeline: face detect --> crop --> classify (whose face) --> notification
* Todo: 
  * collect and annotate more data. 
    * Save images into different folders, the folder name can be used as the label for classifier.
    * Save BBOX label (json format) in the same folder with the same name as the image.
  * notification: send email? save in DB?
* Done: 
  * object detection model built (model, loss function)


# Sudoku Solver

* Take in an sudoku image or detect sudoku puzzle from video, and solve the puzzle.
* Trained a classifier to recognize digits inside the sudoku puzzle.
* The sodoku solver algorithm used a recursion algorithm (call a function inside the function).

