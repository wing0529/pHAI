// Generated by view binder compiler. Do not edit!
package com.example.cameraxx.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
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

public final class MethodPreventBinding implements ViewBinding {
  @NonNull
  private final ConstraintLayout rootView;

  @NonNull
  public final TextView answerTextView;

  @NonNull
  public final ImageButton backButton;

  @NonNull
  public final TextView questionTextView;

  @NonNull
  public final ScrollView scrollView;

  @NonNull
  public final TextView subtitleTextView;

  @NonNull
  public final TextView titleTextView;

  private MethodPreventBinding(@NonNull ConstraintLayout rootView, @NonNull TextView answerTextView,
      @NonNull ImageButton backButton, @NonNull TextView questionTextView,
      @NonNull ScrollView scrollView, @NonNull TextView subtitleTextView,
      @NonNull TextView titleTextView) {
    this.rootView = rootView;
    this.answerTextView = answerTextView;
    this.backButton = backButton;
    this.questionTextView = questionTextView;
    this.scrollView = scrollView;
    this.subtitleTextView = subtitleTextView;
    this.titleTextView = titleTextView;
  }

  @Override
  @NonNull
  public ConstraintLayout getRoot() {
    return rootView;
  }

  @NonNull
  public static MethodPreventBinding inflate(@NonNull LayoutInflater inflater) {
    return inflate(inflater, null, false);
  }

  @NonNull
  public static MethodPreventBinding inflate(@NonNull LayoutInflater inflater,
      @Nullable ViewGroup parent, boolean attachToParent) {
    View root = inflater.inflate(R.layout.method_prevent, parent, false);
    if (attachToParent) {
      parent.addView(root);
    }
    return bind(root);
  }

  @NonNull
  public static MethodPreventBinding bind(@NonNull View rootView) {
    // The body of this method is generated in a way you would not otherwise write.
    // This is done to optimize the compiled bytecode for size and performance.
    int id;
    missingId: {
      id = R.id.answerTextView;
      TextView answerTextView = ViewBindings.findChildViewById(rootView, id);
      if (answerTextView == null) {
        break missingId;
      }

      id = R.id.backButton;
      ImageButton backButton = ViewBindings.findChildViewById(rootView, id);
      if (backButton == null) {
        break missingId;
      }

      id = R.id.questionTextView;
      TextView questionTextView = ViewBindings.findChildViewById(rootView, id);
      if (questionTextView == null) {
        break missingId;
      }

      id = R.id.scrollView;
      ScrollView scrollView = ViewBindings.findChildViewById(rootView, id);
      if (scrollView == null) {
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

      return new MethodPreventBinding((ConstraintLayout) rootView, answerTextView, backButton,
          questionTextView, scrollView, subtitleTextView, titleTextView);
    }
    String missingId = rootView.getResources().getResourceName(id);
    throw new NullPointerException("Missing required view with ID: ".concat(missingId));
  }
}
