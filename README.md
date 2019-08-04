# dino_chrome
bot to play the dinosaur game in chrome

1.first of all the chrome page needs to be in the left hand side of the screen for the screen grab given in the code to work(windows button+left arrow key)

2.dino_keypress is the code for training the data. The script when running will enable us to capture exact moments(using space bar) of our dino jump and store them in the folder called success(which need to be created in the directory).
Whenever the dino isnt jumping, the screengrab captures it and the script puts it in the failure folder(which also needs to be created)

3.We need to trim the training images in both the files and try to make the success and failure approximately equal in number.

4.Using jupyter notebook copy the code in sections separated by the #####'s and run.

5.once finsihed training, save the model.

6.run the dino_final with the trained model.
