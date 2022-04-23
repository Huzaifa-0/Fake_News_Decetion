import 'package:flutter/material.dart';

Widget buildTextField(TextEditingController textController,
        FocusNode textFieldFocusNode, BorderSide borderSide) =>
    Padding(
      padding: const EdgeInsets.all(16.0),
      child: TextField(
        focusNode: textFieldFocusNode,
        style: const TextStyle(color: Colors.white),
        controller: textController,
        maxLines: 13,
        keyboardType: TextInputType.multiline,
        decoration: InputDecoration(
          // labelText: 'Enter your Text here',
          hintText: 'Enter your Text here',
          hintStyle: const TextStyle(color: Colors.white),
          enabledBorder: OutlineInputBorder(
            borderSide: borderSide,
            borderRadius: BorderRadius.circular(10),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(10),
            borderSide: const BorderSide(color: Colors.white, width: 2.5),
          ),
          filled: true,
          fillColor: Colors.black54,
        ),
      ),
    );
