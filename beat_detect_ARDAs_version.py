M = 50
BEAT_THRESHOLD = 2.0        
MIN_BEAT_PERIOD = 500
MIN_ENERGY = 5000          
e_ptr = 0
e_buf = array('L', (0 for i in range(M)))   
sum_energy = 0
c_history = array('f', (0.0 for i in range(M)))
c_ptr = 0
last_beat_time = pyb.millis()   
measured_period = 500           
beat_count = 0                  
BEAT_CONFIRM = 2                

tic = pyb.millis()

while True:
    if audio.buffer_is_filled():

        E = audio.inst_energy()
        audio.reset_buffer()

        sum_energy = sum_energy - e_buf[e_ptr] + E
        e_buf[e_ptr] = E
        e_ptr = (e_ptr + 1) % M
        average_energy = sum_energy / M

        c = E / average_energy
        c_history[c_ptr] = c
        c_ptr = (c_ptr + 1) % M
        BEAT_THRESHOLD = max(c_history) * 0.7

        if (c > BEAT_THRESHOLD):    
            beat_count += 1
        else:
            beat_count = 0

        if average_energy > MIN_ENERGY:
            if (pyb.millis() - tic > MIN_BEAT_PERIOD):
                if beat_count >= BEAT_CONFIRM:  
                    now = pyb.millis()
                    measured_period = now - last_beat_time
                    last_beat_time = now
                    MIN_BEAT_PERIOD = int(measured_period * 0.8)
                    flash()
                    tic = pyb.millis()
                    beat_count = 0