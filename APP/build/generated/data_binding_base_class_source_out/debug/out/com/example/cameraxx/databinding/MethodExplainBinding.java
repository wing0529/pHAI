// Generated by view binder compiler. Do not edit!
package com.example.cameraxx.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.ScrollView;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.example.cameraxx.R;
import java.lang.NullPointerException;
import java.lang.Override;
import java.lang.String;

public final class MethodExplainBinding implements ViewBinding {
  @NonNull
  private final ConstraintLayout rootView;

  @NonNull
  public final ImageButton backButton;

  @NonNull
  public final ImageView resultImageView;

  @NonNull
  public final ScrollView scrollView;

  @NonNull
  public final Button startButton;

  @NonNull
  public final TextView subtitleTextView;

  @NonNull
  public final TextView titleTextView;

  private MethodExplainBinding(@NonNull ConstraintLayout rootView, @NonNull ImageButton backButton,
      @NonNull ImageView resultImageView, @NonNull ScrollView scrollView,
      @NonNull Button startButton, @NonNull TextView subtitleTextView,
      @NonNull TextView titleTextView) {
    this.rootView = rootView;
    this.backButton = backButton;
    this.resultImageView = resultImageView;
    this.scrollView = scrollView;
    this.startButton = startButton;
    this.subtitleTextView = subtitleTextView;
    this.titleTextView = titleTextView;
  }

  @Override
  @NonNull
  public ConstraintLayout getRoot() {
    return rootView;
  }

  @NonNull
  public static MethodExplainBinding inflate(@NonNull LayoutInflater inflater) {
    return inflate(inflater, null, false);
  }

  @NonNull
  public static MethodExplainBinding inflate(@NonNull LayoutInflater inflater,
      @Nullable ViewGroup parent, boolean attachToParent) {
    View root = inflater.inflate(R.layout.method_explain, parent, false);
    if (attachToParent) {
      parent.addView(root);
    }
    return bind(root);
  }

  @NonNull
  public static MethodExplainBinding bind(@NonNull View rootView) {
    // The body of this method is generated in a way you would not otherwise write.
    // This is done to optimize the compiled bytecode for size and performance.
    int id;
    missingId: {
      id = R.id.backButton;
      ImageButton backButton = ViewBindings.findChildViewById(rootView, id);
      if (backButton == null) {
        break missingId;
      }

      id = R.id.resultImageView;
      ImageView resultImageView = ViewBindings.findChildViewById(rootView, id);
      if (resultImageView == null) {
        break missingId;
      }

      id = R.id.scrollView;
      ScrollView scrollView = ViewBindings.findChildViewById(rootView, id);
      if (scrollView == null) {
        break missingId;
      }

      id = R.id.startButton;
      Button startButton = ViewBindings.findChildViewById(rootView, id);
      if (startButton == null) {
        break missingId;
      }

      id = R.id.subtitleTextView;
      TextView subtitleTextView = ViewBindings.findChildViewById(rootView, id);
      if (subtitleTextView == null) {
        break missingId;
      }

      id = R.id.titleTextView;
      TextView titleTextView = ViewBindings.findChildViewById(rootView, id);
      if (titleTextView == null) {
        break missingId;
      }

      return new MethodExplainBinding((ConstraintLayout) rootView, backButton, resultImageView,
          scrollView, startButton, subtitleTextView, titleTextView);
    }
    String missingId = rootView.getResources().getResourceName(id);
    throw new NullPointerException("Missing required view with ID: ".concat(missingId));
  }
}
