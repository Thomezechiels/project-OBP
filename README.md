# project-OBP

## Using the GUI (DSS)
To use the DSS the Graphical User Interface (GUI) can be accessed via a preferred browser, although Google Chrome is recommended. Firstly, the server has to be started by running 'app.py' using Python3. A custom configuration file (in the '.yaml' format) can be passed as an argument. If no configuration file is specified the provided default 'config.yaml' is used. \textbf{Important:} The current directory from which 'app.py' is run should be the 'Project-OBP' directory (the directory that contains 'app.py').  

When the server is up and running, the IP-address on which the GUI can be visited will be shown. The GUI consists of two main screens: 'Parameter Settings' and 'Network Status Dashboard'. 

## Parameter settings
On the 'Parameter Settings' screen all settings relevant to the routing policy, predictive algorithm and simulation can be viewed and edited. The default values are the values specified by the configuration file. 

The Routing Policy setting contains two options: 'Round Robin' and 'Least Connections'. The selection for the predictive algorithm contains four options: 'Random Forest', 'Decision Tree', 'Neural Network', and 'None'. When 'None' is selected no algorithm is used and the number of servers used is fixed for all periods. This number is determined by: $\text{round} (max\_servers / 2)$.

## Network Status Dashboard
For the 'Network Status Dashboard' one of two modes can be chosen: 'Live mode (DSS)' or 'Algorithm Testing Mode'. After choosing one of the two modes it is always possible to switch back to a different mode by clicking the 'Change mode' button on the top of the page. 

### Live Mode (DSS)
In 'Live Mode' the simulation is run in 'real-time'. Here the algorithm suggests the number of servers to use for the next period, rather than choosing it itself. The top three options are calculated and shown alongside with the predicted profit of the upcoming hour. This way the number of servers can be chosen to fit the needs of the system and the user.

To gain insight in the current state of the system, performance metrics of all the servers are displayed in real-time. These metrics contain information on the server id, queue size, performance history, current load, and if the server is active or not. Besides the server metrics, information on the arrivals is also shown on the bottom of the screen.

### Algorithm Testing Mode
Before a algorithm is deployed in the online setting it is good practice to test it and its performance. This can be done in the 'Algorithm Testing Mode'. Currently the models are pre-trained on the simulation settings as specified by the default 'config.yaml' configuration file. How to add new models is explained in the 'Model Training' section. 

After selecting the 'Algorithm Testing Mode' the GUI suggests checking the routing policy, predictive algorithm, and simulation settings. After checking that the right settings are indeed selected, the 'Start simulation' button can be clicked. The simulation will now start running and its progress will be displayed. 

After completion the dashboard will display the results of the simulation. On the left side the total profit is shown alongside with a table containing the statics regarding the profit per period. On the right four graphs are shown. The first graph displays how many servers were used per hour and what the optimum number of servers would have been. The second graph shows the amount of completed requests per hour divided in the categories 'Requests failed', 'Small requests completed', 'Large requests completed'. The third graph is similar to the second graph but displays all requests that were made rather than the ones that were processed. Lastly, the fourth graph displays the average waiting times for both small and large requests. This graph also includes the horizontal lines which represent the threshold for the maximum waiting times. 

After viewing the results the simulation can be re-run with a different algorithm. Change this in the 'Parameter Settings' tab, switch back to the dashboard, click the 'Change mode' button and re-select the 'Algorithm Testing Mode'. 

## Model Training
The models used in the simulation are pre-trained models on the settings as defined by the default configuration file. To load models that have been trained on different simulation settings a separate simulation can be run. To do this 'simulation.py' has to be run with the following arguments: 
  -C --config: Select the configuration file to run the simulation with
  -M --model: Specify which model to use. Currently available models are: 'Decision Tree', 'Random Forest', 'Neural Network'.
  -R --routing: Specify the routing policy to use.
  -O --output: Give the name under which the file should be saved. Models are saved to the 'data/models' directory.
To overwrite the pre-trained models currently available for Decision Tree, Random Forest, and Neural Network use the filenames 'decision\_tree.sav', 'random\_forest.sav', and 'neural\_network.sav' respectively. 
