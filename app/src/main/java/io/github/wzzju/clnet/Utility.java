package io.github.wzzju.clnet;

import android.app.Activity;
import android.content.ContentUris;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.provider.DocumentsContract;
import android.provider.MediaStore;

/**
 * Created by yuchen on 17-12-27.
 */

public class Utility {
    public static float getColorPixel(int pixel, int color) {
        float value = 0;
        float[] meanValue;
        float scaled;
        if (MainActivity.LENET) {
            meanValue = new float[]{0.0f, 0.0f, 0.0f};
            scaled = 255.0f;
        } else {
            meanValue = new float[]{122.562f, 116.538f, 103.885f};
            scaled = 1.0f;
        }
        switch (color) {
            case 0:
                value = ((float) ((pixel >> 16) & 0x000000ff) - meanValue[0]) / scaled;
                break;
            case 1:
                value = ((float) ((pixel >> 8) & 0x000000ff) - meanValue[1]) / scaled;
                break;
            case 2:
                value = ((float) (pixel & 0x000000ff) - meanValue[2]) / scaled;
                break;
        }

        return value;
    }

    //从返回的indent数据获得图片路径
    public static String getPathByData(Activity context, Intent data) {
        if (data == null || data.getData() == null) {
            return null;
        }
        Uri uri = data.getData();
        String[] projection = {MediaStore.Images.Media.DATA};
        Cursor cursor = null;
        switch (uri.getScheme()) {
            case "file":
                cursor = context.getContentResolver().query(uri, projection, null, null, null);
                break;
            case "content":
                switch (uri.getHost()) {
                    case "media":
                        cursor = context.getContentResolver().query(uri, projection, null, null, null);
                        break;
                    case "com.android.providers.media.documents":
                        String wholeID = DocumentsContract.getDocumentId(uri);
                        String id = wholeID.split(":")[1];
                        String sel = MediaStore.Images.Media._ID + "=?";
                        cursor = context.getContentResolver().query(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                                projection, sel, new String[]{id}, null);
                        break;
                    case "com.android.providers.downloads.documents":
                        String dId = DocumentsContract.getDocumentId(uri);
                        Uri contentUri = ContentUris.withAppendedId(Uri.parse("content://downloads/public_downloads"), Long.valueOf(dId));
                        cursor = context.getContentResolver().query(contentUri, projection, null, null, null);
                        break;
                }
                break;
        }
        String path;
        if (cursor != null) {
            cursor.moveToNext();
            int index = cursor.getColumnIndex(MediaStore.Images.Media.DATA);
            path = cursor.getString(index);
            cursor.close();
        } else {
            path = uri.getPath();
        }
        return path;
    }
}
