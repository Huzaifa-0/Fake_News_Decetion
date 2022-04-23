import 'package:flutter/material.dart';

Widget buildTitle() => const Padding(
      padding: EdgeInsets.only(top: 10),
      child: Text(
        'FAKE NEWS DETECTION',
        style: TextStyle(
          color: Colors.white,
          letterSpacing: 1.2,
          fontWeight: FontWeight.w700,
          fontSize: 26,
        ),
      ),
    );
