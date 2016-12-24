# opencv_playground

You might be looking for the oil painting synthesizer.

# painterfun.py

1. install Anaconda for Python 3 environment. <https://www.continuum.io/downloads#windows>
2. install OpenCV for Python
  - if you know what you are doing: `pip install opencv-python`
  - if you are a normal person: Download 
    <http://www.lfd.uci.edu/~gohlke/pythonlibs/f9r7rmd8/opencv_python-3.1.0-cp35-cp35m-win_amd64.whl>
    then `pip install open.....amd64.whl`
3. (must) install threadpool: `pip install threadpool`
4. `ipython -i painterfun.py` to run the program

After step 4 you can start typing code.

by default, `flower.jpg` is loaded. to switch image, modify `painterfun.py`, then exit and run `painterfun.py` again

`r(1)` will try once to put 512 strokes onto the canvas. Then the canvas, original, and the error image will be displayed.

you may execute `r(1)` multiple times.

`r(10)` will try 10 times. that's a lot of strokes!

canvas will be autosaved under /[name of the image file]/ directory between tries. You will be noticed from the CLI.

`hist` variable holds all stroke history.

`repaint(upscale=1)` will paint a new canvas according to the stroke history and return the painted image.

set upscale = 2 to paint a 2x larger image. (will display a smaller version of it, in case it exceeds the size of your screen.

to save the repainted image:

`img = repaint()`, then `cv2.imwrite('abc.png',img)`
