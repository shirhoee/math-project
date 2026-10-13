/**
 * Batch asynchronous image decoding helper.
 * Uses createImageBitmap if supported in the browser, falling back to HTMLImageElement.
 */
export async function decodeFrames(
  base64Frames: string[],
  onProgress?: (loaded: number, total: number) => void
): Promise<(ImageBitmap | HTMLImageElement)[]> {
  const total = base64Frames.length;
  let loaded = 0;

  const decodeSingle = async (dataUrl: string): Promise<ImageBitmap | HTMLImageElement> => {
    try {
      if (typeof window !== 'undefined' && 'createImageBitmap' in window) {
        const response = await fetch(dataUrl);
        const blob = await response.blob();
        const bitmap = await createImageBitmap(blob);
        loaded++;
        onProgress?.(loaded, total);
        return bitmap;
      }
    } catch {
      // Fallback below
    }

    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        loaded++;
        onProgress?.(loaded, total);
        resolve(img);
      };
      img.onerror = reject;
      img.src = dataUrl;
    });
  };

  // Decode concurrently in batches of 10 to balance memory and speed
  const batchSize = 12;
  const results: (ImageBitmap | HTMLImageElement)[] = new Array(total);

  for (let i = 0; i < total; i += batchSize) {
    const chunk = base64Frames.slice(i, i + batchSize);
    const chunkPromises = chunk.map((frame, idx) =>
      decodeSingle(frame).then((decoded) => {
        results[i + idx] = decoded;
      })
    );
    await Promise.all(chunkPromises);
  }

  return results;
}
