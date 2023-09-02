if __name__ == "__main__":
    import cProfile
    from main import main

    pr = cProfile.Profile()
    pr.enable()

    main()

    pr.disable()
    pr.print_stats(sort="time")
