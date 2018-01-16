package io.github.wzzju.clnet;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;

/**
 * Created by yuchen on 17-12-27.
 */

public class CheckPermission {
    public static final int CLNET_PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE = 222;
    public static final int CLNET_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE = 444;
    public static final int CLNET_PERMISSIONS_REQUEST_CAMERA = 666;

    public static boolean checkPermissionRead(
            final Context context) {
        int currentAPIVersion = Build.VERSION.SDK_INT;
        if (currentAPIVersion >= Build.VERSION_CODES.M) {
            if (ContextCompat.checkSelfPermission(context,
                    Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                if (ActivityCompat.shouldShowRequestPermissionRationale(
                        (Activity) context,
                        Manifest.permission.READ_EXTERNAL_STORAGE)) {
                    showDialog("Read external storage", context,
                            Manifest.permission.READ_EXTERNAL_STORAGE, CLNET_PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE);

                } else {
                    ActivityCompat
                            .requestPermissions(
                                    (Activity) context,
                                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                                    CLNET_PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE);
                }
                return false;
            } else {
                return true;
            }

        } else {
            return true;
        }
    }

    public static boolean checkPermissionWrite(
            final Context context) {
        int currentAPIVersion = Build.VERSION.SDK_INT;
        if (currentAPIVersion >= Build.VERSION_CODES.M) {
            if (ContextCompat.checkSelfPermission(context,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                if (ActivityCompat.shouldShowRequestPermissionRationale(
                        (Activity) context,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                    showDialog("Write external storage", context,
                            Manifest.permission.WRITE_EXTERNAL_STORAGE, CLNET_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE);

                } else {
                    ActivityCompat
                            .requestPermissions(
                                    (Activity) context,
                                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                                    CLNET_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE);
                }
                return false;
            } else {
                return true;
            }

        } else {
            return true;
        }
    }

    public static boolean checkPermissionCamera(
            final Context context) {
        int currentAPIVersion = Build.VERSION.SDK_INT;
        if (currentAPIVersion >= Build.VERSION_CODES.M) {
            if (ContextCompat.checkSelfPermission(context,
                    Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                if (ActivityCompat.shouldShowRequestPermissionRationale(
                        (Activity) context,
                        Manifest.permission.CAMERA)) {
                    showDialog("Camera", context,
                            Manifest.permission.CAMERA, CLNET_PERMISSIONS_REQUEST_CAMERA);

                } else {
                    ActivityCompat
                            .requestPermissions(
                                    (Activity) context,
                                    new String[]{Manifest.permission.CAMERA},
                                    CLNET_PERMISSIONS_REQUEST_CAMERA);
                }
                return false;
            } else {
                return true;
            }

        } else {
            return true;
        }
    }

    public static void showDialog(final String msg, final Context context,
                                  final String permission, int permissionCode) {
        AlertDialog.Builder alertBuilder = new AlertDialog.Builder(context);
        alertBuilder.setCancelable(true);
        alertBuilder.setTitle("Permission necessary");
        alertBuilder.setMessage(msg + " permission is necessary");
        alertBuilder.setPositiveButton(android.R.string.yes, (dialog, which) ->
                ActivityCompat.requestPermissions((Activity) context,
                        new String[]{permission},
                        permissionCode)
        );
        AlertDialog alert = alertBuilder.create();
        alert.show();
    }
}
