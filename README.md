# Game Ratings Predictor  
LinkedIn:  [https://www.linkedin.com/in/ganesansantha77/](https://www.linkedin.com/in/ganesansantha77/)
  
Hi DevPeople!  
This is a project originally made by Ganesan

The code is open source and written in Python 3.x but it's also Python 2.x backward compatible.  

The program uses the Kaggle dataset available at https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings and the Python libraries "Pandas" and "Sklearn".  

This project goal is to classifie each video game in the dataset by ESRB rating, to do this we used Logistic Regression, Random Forest and k-NN.  


**Ubuntu, Debian and macOS users:**
First of all, you need **pip** and **Python** installed. 

If you're on Ubuntu or Debian, make sure you have Python Developer installed with `sudo apt-get install python-dev`. 
1. Upgrade pip to the latest version with `pip install --upgrade pip` or `pip2 install --upgrade pip`.
2. Download the repository and open a terminal inside the root.
3. Run `sudo pip install -r requirements.txt` or `sudo pip2 install -r requirements.txt`, it will install all the required packages.
4. Enter `python algoritmo-runner.py` to run the project with without any installation.

You're done!

In addition to this, you can install the program so with `sudo python setup.py install --record files.txt`. Now you can run the whole project with the command line `algoritmo` anywhere in the terminal. You can uninstall it with `cat files.txt | xargs rm -rf`.


**Windows users:**
1. Go to https://www.python.org/downloads/windows/, download and run the .exe
2. Download the `get-pip.py` file from https://bootstrap.pypa.io/get-pip.py
3. Install it with (you might need an administrator command prompt to do this):
	`python get-pip.py`
4. Go to the directory where Python has been installed, now in the Scripts directory make a new file called `local.bat` with the only word `cmd` in it
3. Double click `local.bat` and in the just opened terminal write:
	    
	    pip install numpy
	    pip install scipy
	    pip install pandas
	    pip install -U scikit-learn
4. Run `algoritmo.py` with Python IDLE or the IDE you prefer!
