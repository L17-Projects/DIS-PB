# Allegro Training

This example demonstrates how to train the **DIS-PB model** using **Allegro** with the **NequIP mask branch**.  
In this configuration, **CG_DNA atom forces are not considered during training**.

### Steps

1. **Prepare training data**  
   - Copy the previously prepared `input.data` file into:  
     ```
     allegro/examples/input.data
     ```

2. **Run the training**  
   - From the `allegro/examples` directory, execute:  
     ```
     train.py
     ```  
   - The `train.py` script contains all hyperparameters used for training the **DIS-PB model** as presented in the article.  

3. **Collect results**  
   - After training, results are saved in:  
     ```
     allegro/examples/results
     ```  
   - The key output is `ff.pth`, which contains the **potential of mean force**.  
   - This file should be deployed for running molecular dynamics (MD) simulations.  

---

At the end of this workflow, you will have a trained **DIS-PB model** ready to be used in MD simulations via the `ff.pth` potential file.

