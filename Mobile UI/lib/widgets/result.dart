import 'package:flutter/material.dart';

class ResultContainer extends StatelessWidget {
  const ResultContainer(
      {Key? key, required this.result, required this.isLoading})
      : super(key: key);
  final String? result;
  final bool isLoading;
  @override
  Widget build(BuildContext context) {
    return Container(
      child: isLoading
          ? const Center(
              child: CircularProgressIndicator(color: Colors.white),
            )
          : result == null
              ? const SizedBox.shrink()
              : Text(
                  result == 'real' ? 'REAL!' : 'FAKE!',
                  style: TextStyle(
                    color: result == 'real' ? Colors.green : Colors.red,
                    fontSize: 30,
                    letterSpacing: 1.5,
                    fontWeight: FontWeight.w900,
                  ),
                ),
    );
  }
}
