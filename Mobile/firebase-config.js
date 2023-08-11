import {
    FIREBASE_API_KEY,
    FIREBASE_STORAGE_BUCKET,
    FIREBASE_APP_ID,
    FIREBASE_PROJECT_ID,
    FIREBASE_AUTH_DOMAIN,
  } from "@env";
  import { initializeApp, getApp, getApps } from "firebase/app";
  import {
    getStorage,
    ref,
    uploadBytesResumable,
    getDownloadURL,
    listAll,
  } from "firebase/storage";
  
  // Initialize Firebase
  const firebaseConfig = {
    apiKey: "AIzaSyDUiQPgsw3XzQU4--SdEJQBHCC2p0leis0",
  
    authDomain: "ajdatabase-d8b8f.firebaseapp.com",
  
    projectId: "ajdatabase-d8b8f",
  
    storageBucket: "ajdatabase-d8b8f.appspot.com",
  
    messagingSenderId: "826600023404",
  
    appId: "1:826600023404:web:3646f3bd7ec15fee087f49"
  };
  
  console.log(firebaseConfig);
  
  if (getApps().length === 0) {
    initializeApp(firebaseConfig);
  }
  const fbApp = getApp();
  const fbStorage = getStorage();
  
  const listFiles = async () => {
    const storage = getStorage();
  
    // Create a reference under which you want to list
    const listRef = ref(storage, "images");
  
    // Find all the prefixes and items.
    const listResp = await listAll(listRef);
    return listResp.items;
  };
  
  /**
   *
   * @param {*} uri
   * @param {*} name
   */
  const uploadToFirebase = async (uri, name, onProgress) => {
    const fetchResponse = await fetch(uri);
    const theBlob = await fetchResponse.blob();
  
    const imageRef = ref(getStorage(), `images/${name}`);
  
    const uploadTask = uploadBytesResumable(imageRef, theBlob);
  
    return new Promise((resolve, reject) => {
      uploadTask.on(
        "state_changed",
        (snapshot) => {
          const progress =
            (snapshot.bytesTransferred / snapshot.totalBytes) * 100;
          onProgress && onProgress(progress);
        },
        (error) => {
          // Handle unsuccessful uploads
          console.log(error);
          reject(error);
        },
        async () => {
          const downloadUrl = await getDownloadURL(uploadTask.snapshot.ref);
          resolve({
            downloadUrl,
            metadata: uploadTask.snapshot.metadata,
          });
        }
      );
    });
  };
  
  export { fbApp, fbStorage, uploadToFirebase, listFiles };
  