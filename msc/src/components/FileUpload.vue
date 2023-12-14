<template>
  <div class="upload-container" style="background: url('/audioWaves.gif') no-repeat center center fixed; background-size: cover;">
    <div class="content" style="margin-bottom: 500px;">
      <h1>Music Genre Classification</h1>
      <input type="file" @change="handleFileChange" id="file" class="file-input" accept="audio/*">
      <label for="file" class="file-label">Choose a file</label>
      <p v-if="audioFileName" class="file-name">File Ready: {{ audioFileName }}</p>
      <button @click="analyzeGenre" class="submit-btn" :disabled="isLoading">
        {{ isLoading ? 'Analyzing...' : 'Analyze Genre' }}
      </button>
      <p v-if="isLoading" class="loading">Analyzing, please wait...</p>
      <p v-if="genre" class="genre-result">Predicted Genre: {{ genre }}</p>
      <audio v-if="audioUrl" controls :src="audioUrl" class="audio-player"></audio>
    </div>
  </div>
</template>

<script>
import * as tf from '@tensorflow/tfjs';
import Meyda from 'meyda';

export default {
  data() {
    return {
      model: null,
      audioFile: null,
      genre: null,
      isLoading: false,
      genres: ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'],
      audioFileName: null,
      audioUrl: null,
    };
  },
  methods: {
    // > Method to load the TensorFlow model
    async loadModel() {
      try {
        this.isLoading = true;  // > Indicate that model loading is in progress
        this.model = await tf.loadGraphModel('/model.json'); // > Load the model from the specified path
        this.isLoading = false; // > Update loading status once the model is loaded
      } catch (error) {
        console.error('Error loading model:', error); // > Log any errors during model loading
        this.isLoading = false; // > Ensure loading status is updated in case of an error
      }
    },

    // > Method to handle file selection from the input field
    handleFileChange(event) {
      this.audioFile = event.target.files[0]; // > Store the selected audio file
      this.audioFileName = this.audioFile ? this.audioFile.name : null; // > Store the file name for display
      this.audioUrl = this.audioFile ? URL.createObjectURL(this.audioFile) : null; // > Create a URL for the audio file for playback
    },

    // > Method to analyze the genre of the selected audio file
    async analyzeGenre() {
      if (!this.audioFile) {
        alert('Please upload an audio file first.'); // > Alert if no file is selected
        return;
      }
      this.isLoading = true; // > Indicate that genre analysis is in progress
      this.genre = null; // > Reset the genre to null
      const mfccs = await this.extractMFCCs(this.audioFile); // > Extract MFCCs from the audio file

      console.log('MFCCs shape:', mfccs.length); // > Log the shape of the extracted MFCCs for debugging

      if (mfccs.length !== 13) {
        console.error('Invalid MFCCs shape:', mfccs); // > Check if the extracted features have the correct shape
        this.isLoading = false; // > Update loading status if features are invalid
        return;
      }

      const prediction = this.model.predict(tf.tensor([mfccs])); // > Predict the genre using the model
      const predictionsArray = prediction.dataSync(); // > Convert prediction tensor to JavaScript array

      // > Get the indices of the top 3 predictions
      const topIndices = Array.from(predictionsArray)
        .map((p, i) => ({ p, i }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 3)
        .map(a => a.i);

      // > Map the indices to the corresponding genre names
      this.genre = topIndices.map(index => this.genres[index]);

      this.isLoading = false; // > Update loading status after prediction
    },


    // > Method to extract MFCCs from an audio file
    async extractMFCCs(file) {
      // > Create audio contexts for processing
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const offlineAudioContext = new OfflineAudioContext(
        1, // > Number of channels
        audioContext.sampleRate * 30, // > Buffer size for 30 seconds of audio
        audioContext.sampleRate // > Sample rate
      );

      // > Decode the audio data from the file
      const audioData = await file.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(audioData);

      // > Play the audio file for offline processing
      const bufferSource = offlineAudioContext.createBufferSource();
      bufferSource.buffer = audioBuffer;
      bufferSource.connect(offlineAudioContext.destination);
      bufferSource.start();

      // > Render the audio buffer for offline processing
      const renderedBuffer = await offlineAudioContext.startRendering();

      // > Set up Meyda analyzer to extract features from the rendered buffer
      let mfccs = [];
      let bufferSourceForMeyda = audioContext.createBufferSource();
      bufferSourceForMeyda.buffer = renderedBuffer;
      
      let meydaAnalyzer = Meyda.createMeydaAnalyzer({
        audioContext: audioContext,
        source: bufferSourceForMeyda,
        bufferSize: 2048,
        featureExtractors: ['mfcc'],
        callback: (features) => {
          mfccs.push(features.mfcc); // > Collect MFCCs
        }
      });

      bufferSourceForMeyda.start(0);
      meydaAnalyzer.start();

      // > possible first solution

      // > Collect MFCCs for a short duration
      // > return new Promise(resolve => {
      // >   setTimeout(() => {
      // >     bufferSourceForMeyda.stop();
      // >     meydaAnalyzer.stop();
      // >     console.log("Collected MFCCs:", mfccs); // > Log the collected MFCCs
      // >     resolve(this.averageFeatures(mfccs)); // > Resolve the promise with averaged MFCCs
      // >   }, 10000); // > Collect MFCCs for 10 seconds
      // > });

      // > possible second solution - testing

      return new Promise(resolve => {
        // > Resolve the promise as soon as audio playback ends
        bufferSourceForMeyda.onended = () => {
          console.log("Audio ended, resolving promise early.");
          meydaAnalyzer.stop();
          console.log("Collected MFCCs:", mfccs); // > Log the collected MFCCs
          resolve(this.averageFeatures(mfccs));
        };

        // > Set a timeout based on the audio duration or a maximum limit
        const audioDuration = audioBuffer.duration * 1000; // > Duration in milliseconds
        const timeoutDuration = Math.min(audioDuration, 30000); // > 30 seconds or audio duration, whichever is shorter

        setTimeout(() => {
          console.log("Timeout reached, resolving promise.");
          bufferSourceForMeyda.stop();
          meydaAnalyzer.stop();
          resolve(this.averageFeatures(mfccs));
        }, timeoutDuration);
      });


    },
    // > Method to average the features (MFCCs)
    averageFeatures(featureArray) {
      if (featureArray.length === 0) return []; // > Return an empty array if no features are present

      let summedFeatures = featureArray[0].map(() => 0); // > Initialize an array to store the summed features

      featureArray.forEach(features => {
        features.forEach((feature, index) => {
          summedFeatures[index] += feature; // > Sum features across all frames
        });
      });

      return summedFeatures.map(sum => sum / featureArray.length); // > Average the features by dividing by the number of frames
    },
  },
  mounted() {
    this.loadModel();
  },
};
</script>

<style scoped>
.upload-container {
  position: relative;
  width: 100vw;
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background-size: cover;
  margin: 0;
  padding: 0;
}

.content {
  font-family: Arial, sans-serif;
  text-align: center;
  color: #0db9d7;
  max-width: 500px;
  margin: 0 auto;
}

h1 {
  font-size: 2.5rem;
  margin-bottom: 2rem;
  font-weight: 300;
  color: #fff;
}

.file-input {
  display: none;
}

.file-name, .loading {
    margin-top: 1rem;
    font-size: 1.25rem;
    color: #fff;
  }

  .loading {
    font-style: italic;
  }

.file-label, .submit-btn {
  display: inline-block;
  width: 150px; /* set a fixed width for both buttons */
  padding: 12px 0; /* keep the padding consistent */
  margin: 10px;
  border: none;
  background-color: #0db9d7;
  color: white;
  cursor: pointer;
  border-radius: 30px; /* rounded borders */
  font-size: 16px;
  font-family: Arial, sans-serif;
  transition: background-color 0.3s, transform 0.3s, box-shadow 0.3s;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  text-align: center;
  vertical-align: middle;
  line-height: 1.5; /* center  text vertically */
}

.file-label:hover, .submit-btn:hover {
  background-color: #08a0d7;
  transform: translateY(-2px);
  box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
}

.genre-result {
  margin-top: 1rem;
  font-size: 1.25rem;
  font-weight: bold;
  color: #fff;
}

audio::-webkit-media-controls-panel {
  background-color: #B1D4E0;
}

audio::-webkit-media-controls-play-button {
  background-color: #b1d4e0b3;
  border-radius: 50%;
}

audio::-webkit-media-controls-play-button:hover {
  background-color: #08a0d7;
}

.audio-player {
    margin-bottom: -400px;
    width: 100%;
  }
</style>
