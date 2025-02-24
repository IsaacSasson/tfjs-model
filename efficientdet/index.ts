import * as tf from "@tensorflow/tfjs";
import { Platform } from "react-native";
import { Asset } from "expo-asset";
import { bundleResourceIO } from "@tensorflow/tfjs-react-native";

// For native usage: import model JSON and binary shards
import modelJson from "./model.json";
import shard1 from "./group1-shard1of17.bin";
import shard2 from "./group1-shard2of17.bin";
import shard3 from "./group1-shard3of17.bin";
import shard4 from "./group1-shard4of17.bin";
import shard5 from "./group1-shard5of17.bin";
import shard6 from "./group1-shard6of17.bin";
import shard7 from "./group1-shard7of17.bin";
import shard8 from "./group1-shard8of17.bin";
import shard9 from "./group1-shard9of17.bin";
import shard10 from "./group1-shard10of17.bin";
import shard11 from "./group1-shard11of17.bin";
import shard12 from "./group1-shard12of17.bin";
import shard13 from "./group1-shard13of17.bin";
import shard14 from "./group1-shard14of17.bin";
import shard15 from "./group1-shard15of17.bin";
import shard16 from "./group1-shard16of17.bin";
import shard17 from "./group1-shard17of17.bin";

/**
 * Loads the EfficientDet Lite-4 model.
 * - On **web**, loads the model from a bundled asset (using expo-asset).
 * - On **native** (iOS/Android), loads the model using bundleResourceIO.
 */
export async function loadEfficientdetModel() {
  await tf.ready();

  if (Platform.OS === "web") {
    const modelUrl = "/efficientdet/model.json";
    return await tf.loadGraphModel(modelUrl);
  } else {
    const loadedModel = await tf.loadGraphModel(
      bundleResourceIO(modelJson as any, [
        shard1,
        shard2,
        shard3,
        shard4,
        shard5,
        shard6,
        shard7,
        shard8,
        shard9,
        shard10,
        shard11,
        shard12,
        shard13,
        shard14,
        shard15,
        shard16,
        shard17,
      ])
    );
    return loadedModel;
  }
}
