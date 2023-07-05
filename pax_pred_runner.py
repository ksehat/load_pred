from pax_pred_pretrained_model import pax_pred_pretrained_model
import schedule
import time

# use the schedule.every() method to specify the frequency and time of execution
# for example, to run the hello function every 10 seconds

pax_pred_pretrained_model()
schedule.every(30).minutes.do(pax_pred_pretrained_model)

# use a while loop to keep the program running
while True:
    # run all pending tasks
    schedule.run_pending()
    # wait for one second
    time.sleep(1)