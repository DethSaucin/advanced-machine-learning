<template>
  <div id="app" class="dark-theme">
    <FileUpload />
  </div>
</template>

<script>
import FileUpload from './components/FileUpload.vue';

export default {
  name: 'App',
  components: {
    FileUpload
  }
}
</script>

<style>
#app {
  font-family: 'Arial', sans-serif;
  text-align: center;
  color: #2c3e50;
  height: 100vh; /*  Make app fill the full height of the viewport */
  margin: 0;
  padding: 0;
  background-color: #121212; /* Optional: Set a background color for the entire app if needed */
}
.dark-theme {
  background-color: #121212; /* Dark background for the whole app */
  color: white; /* Light text for contrast */
  height: 100vh; /* Full viewport height */
  margin: 0;
  padding: 0;
}

/* Global styles and resets can go here */
body {
  margin: 0;
  padding: 0;
}
</style>
