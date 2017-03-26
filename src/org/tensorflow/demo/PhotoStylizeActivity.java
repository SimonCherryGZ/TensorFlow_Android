package org.tensorflow.demo;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.webkit.MimeTypeMap;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.GridView;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Toast;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.util.BitmapUtil;
import org.tensorflow.demo.util.FileUtil;
import org.tensorflow.demo.util.MediaScanner;

import java.io.File;
import java.util.ArrayList;

import static org.tensorflow.demo.StylizeActivity.getBitmapFromAsset;

public class PhotoStylizeActivity extends Activity {

    static {
        System.loadLibrary("tensorflow_demo");
    }

    private static final Logger LOGGER = new Logger();

    private static final String MODEL_FILE = "file:///android_asset/stylize_quantized.pb";
    private static final String INPUT_NODE = "input";
    private static final String STYLE_NODE = "style_num";
    private static final String OUTPUT_NODE = "transformer/expand/conv3/conv/Sigmoid";
    private static final int NUM_STYLES = 26;

    // Whether to actively manipulate non-selected sliders so that sum of activations always appears
    // to be 1.0. The actual style input tensor will be normalized to sum to 1.0 regardless.
    private static final boolean NORMALIZE_SLIDERS = true;

    private static final boolean DEBUG_MODEL = false;

    private static final int[] SIZES = {32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};

    // Start at a medium size, but let the user step up through smaller sizes so they don't get
    // immediately stuck processing a large image.
    private int desiredSizeIndex = -1;
    private int desiredSize = 256;

    private final float[] styleVals = new float[NUM_STYLES];

    private int frameNum = 0;

    private Bitmap srcBitmap;
    private Bitmap dstBitmap;

    private TensorFlowInferenceInterface inferenceInterface;

    private int lastOtherStyle = 1;

    private boolean allZero = false;

    private ImageGridAdapter adapter;
    private GridView grid;
    private ImageView ivPhoto;
    private ProgressBar progressBar;
    private View viewMask;

    private StylizeTask stylizeTask = new StylizeTask();


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_photo_stylize);

        grid = (GridView) findViewById(R.id.grid_layout);
        ivPhoto = (ImageView) findViewById(R.id.iv_photo);
        progressBar = (ProgressBar) findViewById(R.id.progress_bar);
        viewMask = findViewById(R.id.view_mask);

        init();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (stylizeTask.getStatus() != AsyncTask.Status.FINISHED) {
            stylizeTask.cancel(true);
            stylizeTask = null;
        }
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) {
            srcBitmap = getBitmapFromImageView(ivPhoto);
            dstBitmap = Bitmap.createBitmap(srcBitmap);
        }
    }

    private void init() {
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
        initListener();
        initStyleGrid();
    }

    private void initListener() {
        findViewById(R.id.btn_reset).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (stylizeTask.getStatus() != AsyncTask.Status.FINISHED) {
                    stylizeTask.cancel(true);
                }
                ivPhoto.setImageBitmap(srcBitmap);
            }
        });

        findViewById(R.id.btn_stylize).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (stylizeTask.getStatus() == AsyncTask.Status.FINISHED) {
                    stylizeTask = new StylizeTask();
                }
                if (stylizeTask.getStatus() != AsyncTask.Status.RUNNING) {
                    progressBar.setVisibility(View.VISIBLE);
                    viewMask.setVisibility(View.VISIBLE);
                    stylizeTask.execute();
                }
            }
        });

        ivPhoto.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startFileManager();
            }
        });
    }

    /**
     * 初始化Style表格
     */
    private void initStyleGrid() {
        adapter = new ImageGridAdapter();
        grid.setAdapter(adapter);
        grid.setOnTouchListener(gridTouchAdapter);

        setStyle(adapter.items[0], 1.0f);
    }

    /**
     * 获取ImageView中的Bitmap
     * @param imageView
     * @return
     */
    private Bitmap getBitmapFromImageView(ImageView imageView) {
        imageView.setDrawingCacheEnabled(true);
        Bitmap bitmap = Bitmap.createBitmap(imageView.getDrawingCache());
        imageView.setDrawingCacheEnabled(false);
        return bitmap;
    }

    /**
     * 打开文件管理器
     */
    private void startFileManager() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(intent, 1024);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (requestCode == 1024) {
                Uri uri = data.getData();
                String path = FileUtil.getFileAbsolutePath(PhotoStylizeActivity.this, uri);
                if (path != null) {
                    srcBitmap = BitmapUtil.decodeSampledBitmapFromFilePath(path, ivPhoto.getWidth(), ivPhoto.getHeight());
                    ivPhoto.setImageBitmap(srcBitmap);
                } else {
                    Toast.makeText(this, "cannot get path", Toast.LENGTH_SHORT).show();
                }
            }
        }
    }

    /**
     * 保存图片后，调用MediaScanner通知系统图库刷新
     * @param fileName
     */
    private void callMediaScanner(String fileName) {
        String root = Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "tensorflow";
        String filePath = root + "/" + fileName;

        MediaScanner mediaScanner = new MediaScanner(getApplicationContext());
        String[] filePaths = new String[]{filePath};
        String[] mimeTypes = new String[]{MimeTypeMap.getSingleton().getMimeTypeFromExtension("png")};
        mediaScanner.scanFiles(filePaths, mimeTypes);
    }

    /**
     * 设置Style
     * @param slider
     * @param value
     */
    private void setStyle(final ImageSlider slider, final float value) {
        slider.setValue(value);

        if (NORMALIZE_SLIDERS) {
            // Slider vals correspond directly to the input tensor vals, and normalization is visually
            // maintained by remanipulating non-selected sliders.
            float otherSum = 0.0f;

            for (int i = 0; i < NUM_STYLES; ++i) {
                if (adapter.items[i] != slider) {
                    otherSum += adapter.items[i].value;
                }
            }

            if (otherSum > 0.0) {
                float highestOtherVal = 0;
                final float factor = otherSum > 0.0f ? (1.0f - value) / otherSum : 0.0f;
                for (int i = 0; i < NUM_STYLES; ++i) {
                    final ImageSlider child = adapter.items[i];
                    if (child == slider) {
                        continue;
                    }
                    final float newVal = child.value * factor;
                    child.setValue(newVal > 0.01f ? newVal : 0.0f);

                    if (child.value > highestOtherVal) {
                        lastOtherStyle = i;
                        highestOtherVal = child.value;
                    }
                }
            } else {
                // Everything else is 0, so just pick a suitable slider to push up when the
                // selected one goes down.
                if (adapter.items[lastOtherStyle] == slider) {
                    lastOtherStyle = (lastOtherStyle + 1) % NUM_STYLES;
                }
                adapter.items[lastOtherStyle].setValue(1.0f - value);
            }
        }

        final boolean lastAllZero = allZero;
        float sum = 0.0f;
        for (int i = 0; i < NUM_STYLES; ++i) {
            sum += adapter.items[i].value;
        }
        allZero = sum == 0.0f;

        // Now update the values used for the input tensor. If nothing is set, mix in everything
        // equally. Otherwise everything is normalized to sum to 1.0.
        for (int i = 0; i < NUM_STYLES; ++i) {
            styleVals[i] = allZero ? 1.0f / NUM_STYLES : adapter.items[i].value / sum;

            if (lastAllZero != allZero) {
                adapter.items[i].postInvalidate();
            }
        }
    }

    /**
     * 对指定Bitmap进行Stylize
     * @param bitmap
     * @return
     */
    private Bitmap stylizeImage(final Bitmap bitmap) {
        desiredSize = bitmap.getWidth();
        int[] intValues = new int[desiredSize * desiredSize];
        float[] floatValues = new float[desiredSize * desiredSize * 3];
        ++frameNum;
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        if (DEBUG_MODEL) {
            // Create a white square that steps through a black background 1 pixel per frame.
            final int centerX = (frameNum + bitmap.getWidth() / 2) % bitmap.getWidth();
            final int centerY = bitmap.getHeight() / 2;
            final int squareSize = 10;
            for (int i = 0; i < intValues.length; ++i) {
                final int x = i % bitmap.getWidth();
                final int y = i / bitmap.getHeight();
                final float val =
                        Math.abs(x - centerX) < squareSize && Math.abs(y - centerY) < squareSize ? 1.0f : 0.0f;
                floatValues[i * 3] = val;
                floatValues[i * 3 + 1] = val;
                floatValues[i * 3 + 2] = val;
            }
        } else {
            for (int i = 0; i < intValues.length; ++i) {
                final int val = intValues[i];
                floatValues[i * 3] = ((val >> 16) & 0xFF) / 255.0f;
                floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / 255.0f;
                floatValues[i * 3 + 2] = (val & 0xFF) / 255.0f;
            }
        }

        // Copy the input data into TensorFlow.
        inferenceInterface.feed(
                INPUT_NODE, floatValues, 1, bitmap.getWidth(), bitmap.getHeight(), 3);
        inferenceInterface.feed(STYLE_NODE, styleVals, NUM_STYLES);

        inferenceInterface.run(new String[] {OUTPUT_NODE}, false);
        inferenceInterface.fetch(OUTPUT_NODE, floatValues);

        for (int i = 0; i < intValues.length; ++i) {
            intValues[i] =
                    0xFF000000
                            | (((int) (floatValues[i * 3] * 255)) << 16)
                            | (((int) (floatValues[i * 3 + 1] * 255)) << 8)
                            | ((int) (floatValues[i * 3 + 2] * 255));
        }

        //bitmap.setPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        Bitmap newBitmap = Bitmap.createBitmap(bitmap);
        newBitmap.setPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        return newBitmap;
    }

    /**
     * 监听Style表格中的ImageSlider的滑动。根据滑动的百分比切换Style。可以多种Style混合。
     */
    private final View.OnTouchListener gridTouchAdapter =
            new View.OnTouchListener() {
                ImageSlider slider = null;

                @Override
                public boolean onTouch(final View v, final MotionEvent event) {
                    switch (event.getActionMasked()) {
                        case MotionEvent.ACTION_DOWN:
                            for (int i = 0; i < NUM_STYLES; ++i) {
                                final ImageSlider child = adapter.items[i];
                                final Rect rect = new Rect();
                                child.getHitRect(rect);
                                if (rect.contains((int) event.getX(), (int) event.getY())) {
                                    slider = child;
                                    slider.setHilighted(true);
                                }
                            }
                            break;

                        case MotionEvent.ACTION_MOVE:
                            if (slider != null) {
                                final Rect rect = new Rect();
                                slider.getHitRect(rect);

                                final float newSliderVal =
                                        (float)
                                                Math.min(
                                                        1.0,
                                                        Math.max(
                                                                0.0, 1.0 - (event.getY() - slider.getTop()) / slider.getHeight()));

                                setStyle(slider, newSliderVal);
                            }
                            break;

                        case MotionEvent.ACTION_UP:
                            if (slider != null) {
                                slider.setHilighted(false);
                                slider = null;
                            }
                            break;

                        default: // fall out

                    }
                    return true;
                }
            };

    /**
     * Style表格的Item控件。
     */
    private class ImageSlider extends ImageView {
        private float value = 0.0f;
        private boolean hilighted = false;

        private final Paint boxPaint;
        private final Paint linePaint;

        public ImageSlider(final Context context) {
            super(context);
            value = 0.0f;

            boxPaint = new Paint();
            boxPaint.setColor(Color.BLACK);
            boxPaint.setAlpha(128);

            linePaint = new Paint();
            linePaint.setColor(Color.WHITE);
            linePaint.setStrokeWidth(10.0f);
            linePaint.setStyle(Paint.Style.STROKE);
        }

        @Override
        public void onDraw(final Canvas canvas) {
            super.onDraw(canvas);
            final float y = (1.0f - value) * canvas.getHeight();

            // If all sliders are zero, don't bother shading anything.
            if (!allZero) {
                canvas.drawRect(0, 0, canvas.getWidth(), y, boxPaint);
            }

            if (value > 0.0f) {
                canvas.drawLine(0, y, canvas.getWidth(), y, linePaint);
            }

            if (hilighted) {
                canvas.drawRect(0, 0, getWidth(), getHeight(), linePaint);
            }
        }

        @Override
        protected void onMeasure(final int widthMeasureSpec, final int heightMeasureSpec) {
            super.onMeasure(widthMeasureSpec, heightMeasureSpec);
            setMeasuredDimension(getMeasuredWidth(), getMeasuredWidth());
        }

        public void setValue(final float value) {
            this.value = value;
            postInvalidate();
        }

        public void setHilighted(final boolean highlighted) {
            this.hilighted = highlighted;
            this.postInvalidate();
        }
    }

    /**
     * Style表格适配器
     */
    private class ImageGridAdapter extends BaseAdapter {
        final ImageSlider[] items = new ImageSlider[NUM_STYLES];
        final ArrayList<Button> buttons = new ArrayList<>();

        {
            final Button sizeButton =
                    new Button(PhotoStylizeActivity.this) {
                        @Override
                        protected void onMeasure(final int widthMeasureSpec, final int heightMeasureSpec) {
                            super.onMeasure(widthMeasureSpec, heightMeasureSpec);
                            setMeasuredDimension(getMeasuredWidth(), getMeasuredWidth());
                        }
                    };
            sizeButton.setText(String.valueOf(desiredSize));
            sizeButton.setOnClickListener(
                    new View.OnClickListener() {
                        @Override
                        public void onClick(final View v) {
                            desiredSizeIndex = (desiredSizeIndex + 1) % SIZES.length;
                            desiredSize = SIZES[desiredSizeIndex];
                            sizeButton.setText(String.valueOf(desiredSize));
                            sizeButton.postInvalidate();
                        }
                    });

            final Button saveButton =
                    new Button(PhotoStylizeActivity.this) {
                        @Override
                        protected void onMeasure(final int widthMeasureSpec, final int heightMeasureSpec) {
                            super.onMeasure(widthMeasureSpec, heightMeasureSpec);
                            setMeasuredDimension(getMeasuredWidth(), getMeasuredWidth());
                        }
                    };
            saveButton.setText("Save");
            saveButton.setOnClickListener(
                    new View.OnClickListener() {
                        @Override
                        public void onClick(final View v) {
                            if (dstBitmap != null) {
                                long timestamp = System.currentTimeMillis();
                                String fileName = "stylized_" + timestamp + ".png";
                                ImageUtils.saveBitmap(dstBitmap, fileName);
                                callMediaScanner(fileName);

                                Toast.makeText(
                                        PhotoStylizeActivity.this,
                                        "Saved image to: /sdcard/tensorflow/" + fileName,
                                        Toast.LENGTH_LONG)
                                        .show();
                            }
                        }
                    });

            buttons.add(sizeButton);
            buttons.add(saveButton);

            for (int i = 0; i < NUM_STYLES; ++i) {
                LOGGER.v("Creating item %d", i);

                if (items[i] == null) {
                    final ImageSlider slider = new ImageSlider(PhotoStylizeActivity.this);
                    final Bitmap bm =
                            getBitmapFromAsset(PhotoStylizeActivity.this, "thumbnails/style" + i + ".jpg");
                    slider.setImageBitmap(bm);

                    items[i] = slider;
                }
            }
        }

        @Override
        public int getCount() {
            return buttons.size() + NUM_STYLES;
        }

        @Override
        public Object getItem(final int position) {
            if (position < buttons.size()) {
                return buttons.get(position);
            } else {
                return items[position - buttons.size()];
            }
        }

        @Override
        public long getItemId(final int position) {
            return getItem(position).hashCode();
        }

        @Override
        public View getView(final int position, final View convertView, final ViewGroup parent) {
            if (convertView != null) {
                return convertView;
            }
            return (View) getItem(position);
        }
    }

    /**
     * 对图片进行Stylize的AsyncTask
     */
    private class StylizeTask extends AsyncTask<Bitmap, Void, Bitmap> {

        @Override
        protected Bitmap doInBackground(Bitmap... params) {
            return stylizeImage(srcBitmap);
        }

        @Override
        protected void onPostExecute(Bitmap bitmap) {
            super.onPostExecute(bitmap);
            progressBar.setVisibility(View.GONE);
            viewMask.setVisibility(View.GONE);
            dstBitmap = bitmap;
            ivPhoto.setImageBitmap(dstBitmap);
        }
    }
}
