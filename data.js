const IMAGE_H = 28;
const IMAGE_W = 28;
const IMAGE_SIZE = IMAGE_H * IMAGE_W;
const N_CLASSES = 10;
const N_DATA = 65000;

const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

export class MnistData {
    constructor() {
        this.isDownloaded = false;
        this.statusElement = document.getElementById('statusText');
        this.progressBar = document.getElementById('progressBar');
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

    async load(nTrain = 40000, nTest = 10000) {
        try {
            this.updateStatus('Loading MNIST data...', 0);
            
            // Make a request for the MNIST sprited image.
            const img = new Image();
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            const imgRequest = new Promise((resolve, reject) => {
                img.crossOrigin = '';
                img.onload = () => {
                    try {
                        this.updateStatus('Processing image data...', 20);
                        
                        img.width = img.naturalWidth;
                        img.height = img.naturalHeight;

                        const datasetBytesBuffer = new ArrayBuffer(N_DATA * IMAGE_SIZE * 4);
                        const chunkSize = 5000;
                        canvas.width = img.width;
                        canvas.height = chunkSize;

                        for (let i = 0; i < N_DATA / chunkSize; i++) {
                            const progress = 20 + (i / (N_DATA / chunkSize) * 30);
                            this.updateStatus(`Processing image chunk ${i + 1}/${Math.ceil(N_DATA / chunkSize)}...`, progress);

                            const datasetBytesView = new Float32Array(
                                datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
                                IMAGE_SIZE * chunkSize);
                            
                            ctx.drawImage(
                                img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
                                chunkSize);

                            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

                            for (let j = 0; j < imageData.data.length / 4; j++) {
                                datasetBytesView[j] = imageData.data[j * 4] / 255;
                            }
                        }
                        this.datasetImages = new Float32Array(datasetBytesBuffer);
                        resolve();
                    } catch (error) {
                        console.error('Error processing image:', error);
                        reject(error);
                    }
                };
                img.onerror = (error) => {
                    console.error('Error loading image:', error);
                    reject(new Error('Failed to load MNIST image data'));
                };
                img.src = MNIST_IMAGES_SPRITE_PATH;
            });

            this.updateStatus('Loading label data...', 50);
            const labelsRequest = fetch(MNIST_LABELS_PATH);
            
            try {
                const [, labelsResponse] = await Promise.all([imgRequest, labelsRequest]);
                
                this.updateStatus('Processing labels...', 70);
                this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

                this.updateStatus('Preparing training and test sets...', 80);
                // Slice the the images and labels into train and test sets.
                this.trainImages = this.datasetImages.slice(0, IMAGE_SIZE * nTrain);
                this.testImages = this.datasetImages.slice(IMAGE_SIZE * nTrain, IMAGE_SIZE * (nTrain + nTest));
                this.trainLabels = this.datasetLabels.slice(0, N_CLASSES * nTrain);
                this.testLabels = this.datasetLabels.slice(N_CLASSES * nTrain, N_CLASSES * (nTrain + nTest));
                
                this.isDownloaded = true;
                this.updateStatus('Data loading complete!', 100);
                
                // Hide progress bar after completion
                setTimeout(() => {
                    if (this.progressBar) {
                        this.progressBar.classList.add('hidden');
                    }
                }, 1000);
                
                console.log('Dataset prepared:', {
                    trainImages: this.trainImages.length,
                    testImages: this.testImages.length,
                    trainLabels: this.trainLabels.length,
                    testLabels: this.testLabels.length
                });
            } catch (error) {
                console.error('Error loading labels:', error);
                this.updateStatus('Error loading MNIST data. Please refresh to try again.');
                throw error;
            }
        } catch (error) {
            console.error('Error in load():', error);
            this.updateStatus('Failed to load MNIST data. Please refresh to try again.');
            throw error;
        }
    }

    /**
    * Get all training data as a data tensor and a labels tensor.
    *
    * @returns
    *   x_train: The data tensor, of shape `[numTrainExamples, 28, 28, 1]`.
    *   y_train: The one-hot encoded labels tensor, of shape
    *     `[numTrainExamples, 10]`.
    */
    getTrainData() {
        const x_train = tf.tensor4d(
                       this.trainImages,
                       [this.trainImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
        const y_train = tf.tensor2d(
                           this.trainLabels, [this.trainLabels.length / N_CLASSES, N_CLASSES]);
        return [x_train, y_train];
    }

    /**
    * Get all test data as a data tensor a a labels tensor.
    *
    * @param {number} numExamples Optional number of examples to get. If not
    *     provided,
    *   all test examples will be returned.
    * @returns
    *   x_test: The data tensor, of shape `[numTestExamples, 28, 28, 1]`.
    *   y_test: The one-hot encoded labels tensor, of shape
    *     `[numTestExamples, 10]`.
    */
    getTestData(numExamples) {
        let x_test = tf.tensor4d(
                     this.testImages,
                     [this.testImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
        let y_test = tf.tensor2d(
                         this.testLabels, [this.testLabels.length / N_CLASSES, N_CLASSES]);

        if (numExamples != null) {
            x_test = x_test.slice([0, 0, 0, 0], [numExamples, IMAGE_H, IMAGE_W, 1]);
            y_test = y_test.slice([0, 0], [numExamples, N_CLASSES]);
        }
        return [x_test, y_test];
    }
}