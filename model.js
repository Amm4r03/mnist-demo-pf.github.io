const model = tf.sequential();

export class MNISTModel {
    constructor() {
        this.statusElement = document.getElementById('statusText');
        this.progressBar = document.getElementById('progressBar');
        this.createModel();
    }

    updateStatus(message, progress = null) {
        console.log(message);
        if (this.statusElement) {
            this.statusElement.textContent = message;
        }
        if (progress !== null && this.progressBar) {
            this.progressBar.classList.remove('hidden');
            this.progressBar.value = progress;
        }
    }

    createModel() {
        try {
            this.model = tf.sequential();
            
            // Input layer with flatten operation
            this.model.add(tf.layers.flatten({inputShape: [28, 28, 1]}));
            
            // Hidden layers
            this.model.add(tf.layers.dense({
                units: 128,
                activation: 'relu',
                kernelInitializer: 'varianceScaling'
            }));
            this.model.add(tf.layers.dropout(0.3));

            this.model.add(tf.layers.dense({
                units: 64,
                activation: 'relu',
                kernelInitializer: 'varianceScaling'
            }));
            this.model.add(tf.layers.dropout(0.3));

            // Output layer
            this.model.add(tf.layers.dense({
                units: 10,
                activation: 'softmax',
                kernelInitializer: 'varianceScaling'
            }));

            // Compile the model
            this.model.compile({
                optimizer: 'adam',
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            });

            console.log('Model created successfully');
            console.log(this.model.summary());
        } catch (error) {
            console.error('Error creating model:', error);
            this.updateStatus('Error creating model. Check console for details.');
            throw error;
        }
    }

    async train(xTrain, yTrain, epochs = 5, batchSize = 128) {
        try {
            this.updateStatus('Starting model training...', 0);
            
            const totalSteps = epochs * Math.ceil(xTrain.shape[0] / batchSize);
            let currentStep = 0;

            const history = await this.model.fit(xTrain, yTrain, {
                epochs: epochs,
                batchSize: batchSize,
                shuffle: true,
                validationSplit: 0.1,
                callbacks: {
                    onBatchEnd: async (batch, logs) => {
                        currentStep++;
                        const progress = (currentStep / totalSteps) * 100;
                        this.updateStatus(
                            `Training... Epoch ${currentStep} of ${totalSteps}`,
                            progress
                        );
                    },
                    onEpochEnd: async (epoch, logs) => {
                        console.log(
                            `Epoch ${epoch + 1}/${epochs}:`,
                            `loss = ${logs.loss.toFixed(4)},`,
                            `accuracy = ${logs.acc.toFixed(4)},`,
                            `validation_loss = ${logs.val_loss.toFixed(4)},`,
                            `validation_accuracy = ${logs.val_acc.toFixed(4)}`
                        );
                    }
                }
            });

            this.updateStatus('Training complete!', 100);
            setTimeout(() => {
                if (this.progressBar) {
                    this.progressBar.classList.add('hidden');
                }
            }, 1000);

            return history;
        } catch (error) {
            console.error('Error during training:', error);
            this.updateStatus('Error during training. Check console for details.');
            throw error;
        }
    }

    async predict(input) {
        try {
            console.log('Starting prediction...', input.shape);
            
            // Ensure input is a tensor and reshape if needed
            if (!(input instanceof tf.Tensor)) {
                input = tf.tensor(input);
            }
            
            // Add batch dimension if needed - this is the key fix
            if (input.shape.length === 3) {
                input = input.expandDims(0);
                console.log('Expanded input shape:', input.shape);
            }
            
            const prediction = await this.model.predict(input);
            console.log('Prediction made successfully');
            
            return prediction;
        } catch (error) {
            console.error('Error during prediction:', error);
            this.updateStatus('Error making prediction. Check console for details.');
            throw error;
        }
    }
}